model_name=Llama-3-8B-Instruct-Gradient-1048k

batch_size=1
prefill_len_list=(4000)
decode_len=128
precision=w8a8kv8
kv_quant_granularity=per_tensor

static_sparsity=0.3
sparse_prefill_mode=1

sparse_decode_mode=1
dynamic_attn_budget=4096
dynamic_select_interval=4
sub_chunk_per_block=4

device_id=2

for prefill_len in "${prefill_len_list[@]}"; do
    bash scripts/lserve_benchmark/benchmark.sh \
        QServe-benchmarks/$model_name \
        attn_patterns/$model_name \
        $batch_size $prefill_len $decode_len \
        $precision $kv_quant_granularity \
        $static_sparsity $sparse_prefill_mode \
        $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block \
        $device_id
done
