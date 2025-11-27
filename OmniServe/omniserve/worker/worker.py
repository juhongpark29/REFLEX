# original file: https://github.com/vllm-project/vllm/blob/main/vllm/worker/worker.py
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

"""A GPU worker class."""

import os
from typing import Dict, List, Optional

import torch
import torch.distributed

from omniserve.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
)
from omniserve.sequence import SamplerOutput, SequenceGroupMetadata
from omniserve.utils.utils import STR_DTYPE_TO_TORCH_DTYPE
from omniserve.worker.model_runner import ModelRunner
from omniserve.modeling.layers.ctx_attn.block_table_utils import pad_block_tables, get_layer_block_tables


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        kv_cache_dtype: Optional[torch.dtype] = torch.int8,
        is_driver_worker: bool = False,
        precision: str = "w4a8",
        kv_cache_config: Optional[Dict] = None,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."
        self.kv_cache_dtype = kv_cache_dtype
        self.model_runner = None
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
        self.precision = precision
        self.kv_cache_config = kv_cache_config

    def init_model(self, cupy_port: Optional[int] = None) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            # _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        # init_distributed_environment(
        #     self.parallel_config, self.rank, cupy_port, self.distributed_init_method
        # )
        # Initialize the model.
        # set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def init_cache_engine(
        self,
        cache_config: CacheConfig,
        quant_path: Optional[str] = None,
        group_size: int = -1,
    ) -> None:
        # Cache engine is initialized inside the model runner.
        self.cache_config = cache_config
        self.model_runner = ModelRunner(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            self.device_config,
            kv_cache_dtype=self.kv_cache_dtype,
            is_driver_worker=True,
            precision=self.precision,
            kv_cache_config=self.kv_cache_config,
            quant_path=quant_path,
            group_size=group_size,
        )
        self.cache_engine = self.model_runner.cache_engine

    def init_block_tables(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        sliding_window: int = None,
    ) -> None:
        # Prepare block tables prior to model execution (non-ifb mode).
        block_size = self.cache_config.block_size
        device = self.device_config.device

        self.retrieval_block_tables: List[List[int]] = []
        self.streaming_block_tables: List[List[int]] = []
        kv_scales_ptrs: List[torch.Tensor] = []

        kv_scale_layer_offsets = (
            torch.arange(
                self.model_config.get_num_layers(self.parallel_config)
            ).unsqueeze(0)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        kv_scale_kv_offsets = (
            torch.arange(2).unsqueeze(1)
            * self.model_config.get_num_layers(self.parallel_config)
            * self.model_config.max_model_len
            * self.model_config.get_num_kv_heads(self.parallel_config)
        )
        kv_scale_offsets = (kv_scale_layer_offsets + kv_scale_kv_offsets).to(
            self.device_config.device
        )

        for seq_group_metadata in seq_group_metadata_list:
            assert (
                seq_group_metadata.is_prompt
            )  # We only do kv block table initialization once
            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                retrieval_block_table = seq_group_metadata.retrieval_block_tables[seq_id]
                self.retrieval_block_tables.append(retrieval_block_table)
                
                if self.model_config.sp_attn_config.sparse_kv_cache_enabled():
                    sink_size, local_size = self.model_config.sp_attn_config.get_dec_sink_size(), self.model_config.sp_attn_config.get_dec_sink_size()
                    sink_block, local_block = self.model_config.sp_attn_config.get_dec_sink_block_num(), self.model_config.sp_attn_config.get_dec_local_block_num()
                    streaming_block_table = seq_group_metadata.streaming_block_tables[seq_id]
                    if streaming_block_table is not None:
                        length = len(streaming_block_table)
                        if length > sink_block + local_block:
                            streaming_block_table = streaming_block_table[:sink_block] + streaming_block_table[-local_block:]
                        self.streaming_block_tables.append(streaming_block_table)
                
        # print(f"[Worker init_block_tables]")
        # print(f"retrieval_block_tables: {self.retrieval_block_tables}")
        # print(f"streaming_block_tables: {self.streaming_block_tables}")
                  
        (
            self.retrieval_block_tables, 
            self.streaming_block_tables, 
            self.max_retrieval_block_table_len, 
            self.max_streaming_block_table_len
        ) = pad_block_tables(
            retrieval_block_tables=self.retrieval_block_tables,                
            streaming_block_tables=self.streaming_block_tables, 
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device=device
        )
        
        self.layer_kv_scales = None  # torch.cat(kv_scales_ptrs, dim=2)

        layers = self.model_config.get_num_layers(self.parallel_config)
        # print(f"[Worker init_block_tables]")
        (
            self.layer_retrieval_block_tables, 
            self.layer_streaming_block_tables
        ) = get_layer_block_tables(
            cache_engine=self.cache_engine,
            layers=layers,
            cache_config=self.cache_config,
            retrieval_block_tables=self.retrieval_block_tables,
            streaming_block_tables=self.streaming_block_tables,
            sparse_kv_cache_enabled=self.model_config.sp_attn_config.sparse_kv_cache_enabled(),
            device=device
        )

    def warm_up_model(self) -> None:
        pass

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        retrieval_blocks_to_swap_in: Optional[Dict[int, int]] = None,
        streaming_blocks_to_swap_in: Optional[Dict[int, int]] = None,
        retrieval_blocks_to_swap_out: Optional[Dict[int, int]] = None,
        streaming_blocks_to_swap_out: Optional[Dict[int, int]] = None,
        retrieval_blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        streaming_blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        ifb_mode: bool = True,
    ) -> Optional[SamplerOutput]:
        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        if ifb_mode:
            # print("$$$$$$$$$$Worker IFB mode")
            output = self.model_runner.execute_model(
                seq_group_metadata_list, self.gpu_cache, ifb_mode=ifb_mode
            )
        else:
            # print("$$$$$$$$$$Worker non-IFB mode")
            output = self.model_runner.execute_model(
                seq_group_metadata_list,
                self.gpu_cache,
                ifb_mode=ifb_mode,
                layer_retrieval_block_tables=self.layer_retrieval_block_tables,
                layer_streaming_block_tables=self.layer_streaming_block_tables,
                max_retrieval_block_table_len=self.max_retrieval_block_table_len,
                max_streaming_block_table_len=self.max_streaming_block_table_len,
                layer_kv_scales=self.layer_kv_scales,
            )
        return output
