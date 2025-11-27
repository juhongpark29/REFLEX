import torch
from omniserve.config import (
    CacheConfig,
    ModelConfig,
)
from typing import Dict, List, Optional
from omniserve.utils.utils import STR_DTYPE_TO_TORCH_DTYPE
from omniserve.worker.cache_engine import CacheEngine
from typing import Dict, List, Optional, Tuple, Union

_sizeof = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int8: 1}

def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))

def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


def pad_block_tables(
    retrieval_block_tables: List[List[int]],
    streaming_block_tables: List[List[int]],
    sparse_kv_cache_enabled: bool,
    device: torch.device,
):
    max_retrieval_block_table_len = max(
        len(block_table) for block_table in retrieval_block_tables
    )
    retrieval_block_tables = _make_tensor_with_pad(
        retrieval_block_tables,
        max_len=max_retrieval_block_table_len,
        pad=0,
        dtype=torch.long,
        device=device,
    )
    
    if sparse_kv_cache_enabled:
        max_streaming_block_table_len = max(
            len(block_table) for block_table in streaming_block_tables
        )
        streaming_block_tables = _make_tensor_with_pad(
            streaming_block_tables,
            max_len=max_streaming_block_table_len,
            pad=0,
            dtype=torch.long,
            device=device,
        )
    else:
        max_streaming_block_table_len = 0
        
    return retrieval_block_tables, streaming_block_tables, max_retrieval_block_table_len, max_streaming_block_table_len


def get_layer_block_tables(
      cache_engine: CacheEngine,
      layers: int,
      cache_config: CacheConfig,
      retrieval_block_tables: List[List[int]],
      streaming_block_tables: List[List[int]],
      sparse_kv_cache_enabled: bool,
      device: torch.device,
):
    layer_retrieval_block_tables = []
    layer_streaming_block_tables = []
    
    for l in range(layers):
        base_retrieval_key_ptrs = cache_engine.get_retrieval_k_gpu_cache_ptr(l)
        base_retrieval_value_ptrs = cache_engine.get_retrieval_v_gpu_cache_ptr(l)
        retrieval_block_offsets_k = (
            retrieval_block_tables
            * cache_engine.get_retrieval_gpu_num_bytes_per_block_k(l)
            * _sizeof[
                STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
            ]
        )
        retrieval_block_offsets_v = (
            retrieval_block_tables
            * cache_engine.get_retrieval_gpu_num_bytes_per_block_v(l)
            * _sizeof[
                STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
            ]
        )
        retrieval_key_ptrs = base_retrieval_key_ptrs + retrieval_block_offsets_k
        retrieval_value_ptrs = base_retrieval_value_ptrs + retrieval_block_offsets_v
        layer_retrieval_block_tables.append(
            torch.cat((retrieval_key_ptrs.unsqueeze(1), retrieval_value_ptrs.unsqueeze(1)), dim=1).to(device)
        )
        
        if sparse_kv_cache_enabled:
            base_streaming_key_ptrs = cache_engine.get_streaming_k_gpu_cache_ptr(l)
            base_streaming_value_ptrs = cache_engine.get_streaming_v_gpu_cache_ptr(l)
            streaming_block_offsets_k = (
                streaming_block_tables
                * cache_engine.get_streaming_gpu_num_bytes_per_block_k(l)
                * _sizeof[
                    STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
                ]
            )
            streaming_block_offsets_v = (
                streaming_block_tables
                * cache_engine.get_streaming_gpu_num_bytes_per_block_v(l)
                * _sizeof[
                    STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
                ]
            )
            streaming_key_ptrs = base_streaming_key_ptrs + streaming_block_offsets_k
            streaming_value_ptrs = base_streaming_value_ptrs + streaming_block_offsets_v
            layer_streaming_block_tables.append(
                torch.cat((streaming_key_ptrs.unsqueeze(1), streaming_value_ptrs.unsqueeze(1)), dim=1).to(device)
            )
        else:
            layer_streaming_block_tables.append(None)

    return layer_retrieval_block_tables, layer_streaming_block_tables