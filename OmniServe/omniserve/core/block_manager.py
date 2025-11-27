# original file: https://github.com/vllm-project/vllm/blob/main/vllm/core/block_manager.py
# modified by: Haotian Tang and Shang Yang
# @article{lin2024qserve,
#   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
#   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
#   year={2024}
# }
# @article{yang2025lserve,
#   title={LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention},
#   author={Yang*, Shang and Guo*, Junxian and Tang, Haotian and Hu, Qinghao and Xiao, Guangxuan and Tang, Jiaming and Lin, Yujun and Liu, Zhijian and Lu, Yao and Han, Song},
#   year={2025}
# }
"""A block manager that manages token blocks."""
import enum
from typing import Dict, List, Optional, Set, Tuple

from omniserve.block import BlockTable, PhysicalTokenBlock
from omniserve.sequence import Sequence, SequenceGroup, SequenceStatus
from omniserve.utils.utils import Device
from omniserve.attn_config import SpAttnConfig


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(
                device=device, block_number=i, block_size=block_size
            )
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """

    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BaseBlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sink_local_blocks: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.streaming_enabled = False
        if sink_local_blocks is not None:
            self.streaming_enabled = True
            sink_blocks, local_blocks = sink_local_blocks
            self.block_sink_window = sink_blocks
            self.block_local_window = local_blocks

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size, num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)
        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(
        self, seq_group: SequenceGroup, ifb_mode: bool, init_num_blocks: int = None
    ) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = len(seq.logical_token_blocks)
        if init_num_blocks is not None:
            assert (
                ifb_mode == False
            )  # In non-ifb mode, we initialize the block tables all at once
            num_required_blocks = max(init_num_blocks, num_required_blocks)

        if seq_group.prefix is not None and seq_group.prefix.allocated:
            num_required_blocks -= seq_group.prefix.get_num_blocks()

        if self.streaming_enabled:
            num_required_blocks = min(num_required_blocks, self.block_sink_window + self.block_local_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(
        self, seq_group: SequenceGroup, ifb_mode: bool, init_num_blocks: int = None
    ) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)
        if init_num_blocks is not None:
            assert (
                ifb_mode == False
            )  # In non-ifb mode, we initialize the block tables all at once
            num_prompt_blocks = max(init_num_blocks, num_prompt_blocks)

        block_table: BlockTable = []
        prefix_block_table: BlockTable = []
        num_prefix_blocks = 0

        prefix = seq_group.prefix
        if prefix is not None and prefix.allocated:
            # Prefix has already been allocated. Use the existing block table.
            num_prompt_blocks -= prefix.get_num_blocks()
            for block in prefix.block_table:
                block.ref_count += seq_group.num_seqs()
                block_table.append(block)
        # print("################# num_prompt_blocks", num_prompt_blocks)
        for logical_idx in range(num_prompt_blocks):
            if (
                self.streaming_enabled 
                and logical_idx >= self.block_sink_window + self.block_local_window
            ):
                block = block_table[self.block_sink_window + (logical_idx - self.block_sink_window) % self.block_local_window]
            else:
                block = self.gpu_allocator.allocate()
            # Set the reference counts of the token blocks.
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)
            # print("logical_idx", logical_idx, "block", block, "block_table", block_table)

        if prefix is not None and not prefix.allocated:
            # Allocate blocks for the prefix, we will compute the prefix's
            # KV cache in this run.
            num_prefix_blocks = prefix.get_num_blocks()
            prefix_block_table = block_table[:num_prefix_blocks]
            for block in prefix_block_table:
                block.ref_count += 1
            prefix.set_block_table(prefix_block_table)

        # Assign the block table for each sequence.
        # if num_prompt_blocks > self.block_sink_window + self.block_local_window:
        #     block_table = block_table[:self.block_sink_window] + block_table[-self.block_local_window:]
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            if (
                self.streaming_enabled 
                and len(block_table) >= self.block_sink_window + self.block_local_window
            ):
                # re-use a block
                block_table.append(
                    block_table[self.block_sink_window + (len(block_table) - self.block_sink_window) % self.block_local_window]
                )
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.
                block = self.gpu_allocator.allocate()
                block_table.append(block)
                return None

        # We want to append the token to the last physical block.
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
        self, seq_group: SequenceGroup
    ) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        if seq_group.prefix is not None:
            # make sure to swap in the prefix first
            assert seq_group.prefix.allocated and seq_group.prefix.computed

        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            if seq_group.prefix is not None:
                for block in seq_group.prefix.block_table:
                    new_block_table.append(block)
                    block.ref_count += 1

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if (
                    seq_group.prefix is not None
                    and gpu_block in seq_group.prefix.block_table
                ):
                    # NOTE: We do not swap out the prefix blocks for now.
                    self.gpu_allocator.free(gpu_block)
                    continue

                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in set(block_table):
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
    
    

class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_retrieval_gpu_blocks: int,
        num_retrieval_cpu_blocks: int,
        num_streaming_gpu_blocks: int,
        num_streaming_cpu_blocks: int,
        sp_attn_config: SpAttnConfig,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_retrieval_gpu_blocks = num_retrieval_gpu_blocks
        self.num_total_retrieval_cpu_blocks = num_retrieval_cpu_blocks
        self.num_total_streaming_gpu_blocks = num_streaming_gpu_blocks
        self.num_total_streaming_cpu_blocks = num_streaming_cpu_blocks
        self.block_sliding_window = None
        self.watermark = watermark
        assert watermark >= 0.0
        self.sparse_kv_cache_enabled = sp_attn_config.sparse_kv_cache_enabled()
        self.retrieval_blockspace_manager = BaseBlockSpaceManager(
            block_size, 
            num_retrieval_gpu_blocks, 
            num_retrieval_cpu_blocks, 
            watermark
        )
        self.streaming_blockspace_manager = None
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager = BaseBlockSpaceManager(
                block_size, 
                num_streaming_gpu_blocks, 
                num_streaming_cpu_blocks, 
                watermark, 
                (sp_attn_config.get_dec_sink_block_num(), sp_attn_config.get_dec_local_block_num()) # add by JXGuo: one more local block for future design
            )

    def can_allocate(
        self, seq_group: SequenceGroup, ifb_mode: bool, init_num_blocks: int = None
    ) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        
        retrieval_status = self.retrieval_blockspace_manager.can_allocate(
            seq_group, ifb_mode, init_num_blocks
        )
        if not self.sparse_kv_cache_enabled:
            return retrieval_status
        streaming_status = self.streaming_blockspace_manager.can_allocate(
            seq_group, ifb_mode, init_num_blocks
        )
        if retrieval_status == AllocStatus.NEVER or streaming_status == AllocStatus.NEVER:
            return AllocStatus.NEVER
        if retrieval_status == AllocStatus.OK and streaming_status == AllocStatus.OK:
            return AllocStatus.OK
        return AllocStatus.LATER
    

    def allocate(
        self, seq_group: SequenceGroup, ifb_mode: bool, init_num_blocks: int = None
    ) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        self.retrieval_blockspace_manager.allocate(
            seq_group, ifb_mode, init_num_blocks
        )
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager.allocate(
                seq_group, ifb_mode, init_num_blocks
            )

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        retrieval_status = self.retrieval_blockspace_manager.can_append_slot(seq_group)
        if not self.sparse_kv_cache_enabled:
            return retrieval_status
        streaming_status = self.streaming_blockspace_manager.can_append_slot(seq_group)
        return retrieval_status and streaming_status

    def append_slot(self, seq: Sequence) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Allocate a physical slot for a new token."""
        
        retrieval_result = self.retrieval_blockspace_manager.append_slot(seq)
        if not self.sparse_kv_cache_enabled:
            return retrieval_result, None
        streaming_result = self.streaming_blockspace_manager.append_slot(seq)
        return retrieval_result, streaming_result


    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        self.retrieval_blockspace_manager.fork(parent_seq, child_seq)
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager.fork(parent_seq, child_seq)
        # src_block_table = self.block_tables[parent_seq.seq_id]
        # self.block_tables[child_seq.seq_id] = src_block_table.copy()
        # for block in src_block_table:
        #     block.ref_count += 1

    def _get_physical_blocks(
        self, seq_group: SequenceGroup
    ) -> Tuple[List[PhysicalTokenBlock], Optional[List[PhysicalTokenBlock]]]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        retrieval_blocks = self.retrieval_blockspace_manager._get_physical_blocks(seq_group)
        streaming_blocks = None
        if self.sparse_kv_cache_enabled:
            streaming_blocks = self.streaming_blockspace_manager._get_physical_blocks(seq_group)
        return retrieval_blocks, streaming_blocks

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        retrieval_result = self.retrieval_blockspace_manager.can_swap_in(seq_group)
        if not self.sparse_kv_cache_enabled:
            return retrieval_result
        streaming_result = self.streaming_blockspace_manager.can_swap_in(seq_group)
        return retrieval_result and streaming_result


    def swap_in(self, seq_group: SequenceGroup) -> Tuple[Dict[int, int], Optional[Dict[int, int]]]:
        # CPU block -> GPU block.
        retrieval_mapping = self.retrieval_blockspace_manager.swap_in(seq_group)
        streaming_mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        if self.sparse_kv_cache_enabled:
            streaming_mapping = self.streaming_blockspace_manager.swap_in(seq_group)
        return retrieval_mapping, streaming_mapping


    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        retrieval_result = self.retrieval_blockspace_manager.can_swap_out(seq_group)
        if not self.sparse_kv_cache_enabled:
            return retrieval_result
        streaming_result = self.streaming_blockspace_manager.can_swap_out(seq_group)
        return retrieval_result and streaming_result

    def swap_out(self, seq_group: SequenceGroup) -> Tuple[Dict[int, int], Optional[Dict[int, int]]]:
        # GPU block -> CPU block.
        retrieval_mapping = self.retrieval_blockspace_manager.swap_out(seq_group)
        streaming_mapping = None
        if self.sparse_kv_cache_enabled:
            streaming_mapping = self.streaming_blockspace_manager.swap_out(seq_group)
        return retrieval_mapping, streaming_mapping

    def _free_block_table(self, retrieval_block_table: BlockTable, streaming_block_table: Optional[BlockTable]) -> None:
        self.retrieval_blockspace_manager._free_block_table(retrieval_block_table)
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager._free_block_table(streaming_block_table)

    def free(self, seq: Sequence) -> None:
        self.retrieval_blockspace_manager.free(seq)
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager.free(seq)

    def reset(self) -> None:
        self.retrieval_blockspace_manager.reset()
        if self.sparse_kv_cache_enabled:
            self.streaming_blockspace_manager.reset()

    def get_retrieval_block_table(self, seq: Sequence) -> List[int]:
        retrieval_block_table = self.retrieval_blockspace_manager.get_block_table(seq)
        return retrieval_block_table
    
    def get_streaming_block_table(self, seq: Sequence) -> Optional[List[int]]:
        if self.sparse_kv_cache_enabled:
            streaming_block_table = self.streaming_blockspace_manager.get_block_table(seq)
            return streaming_block_table
        return None
        
    
    def get_retrieval_num_free_gpu_blocks(self) -> int:
        return self.retrieval_blockspace_manager.get_num_free_gpu_blocks()

    def get_streaming_num_free_cpu_blocks(self) -> Optional[int]:
        if self.sparse_kv_cache_enabled:
            return self.streaming_blockspace_manager.get_num_free_cpu_blocks()
        return None

    
    
    
    