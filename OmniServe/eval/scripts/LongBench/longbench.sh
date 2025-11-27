cd eval/LongBench
mkdir -p logs

base_model=$1
model_path=$2
attn_pattern_path=$3
task=$4
static_sparsity=$5
sparse_prefill_mode=$6
precision=$7
kv_quant_granularity=$8
sparse_decode_mode=$9
dynamic_sparse_token_budget=${10}
selector_update_interval=${11}
sub_chunk_per_block=${12}
device=${13}

echo "[INFO] Running LongBench for task: $task, sparse_prefill_mode: $sparse_prefill_mode, static_sparsity: $static_sparsity"

# Decoding attention experiment parameters (from environment variables)

export EXPERIMENT_MODE=${EXPERIMENT_MODE:-0}
export MULTIPLE_VALUE=${MULTIPLE_VALUE:-2}
export MULTIPLE_PERCENTAGE=${MULTIPLE_PERCENTAGE:-70}

# Fix PyTorch CUDA memory allocation error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "[DEBUG] EXPERIMENT_MODE: $EXPERIMENT_MODE"
echo "[DEBUG] MULTIPLE_VALUE: $MULTIPLE_VALUE"
echo "[DEBUG] MULTIPLE_PERCENTAGE: $MULTIPLE_PERCENTAGE"

ctx_sink_token=128
ctx_local_token=4096
dec_sink_token=128
dec_local_token=256

ckpt_name=$(basename "$model_path")

# Process experiment parameters (similar to niah_test.sh)
experiment_mode=$EXPERIMENT_MODE
multiple_value=$MULTIPLE_VALUE
multiple_percentage=$MULTIPLE_PERCENTAGE

# Clean up values (remove any '=' prefix)
experiment_mode=$(echo "$experiment_mode" | sed 's/.*=//g')
multiple_value=$(echo "$multiple_value" | sed 's/.*=//g')
multiple_percentage=$(echo "$multiple_percentage" | sed 's/.*=//g')

# Calculate experiment suffix
if [ "$experiment_mode" == "4" ]; then
    percentage="$multiple_percentage"
    exp_suffix="exp${experiment_mode}_G${multiple_value}${percentage}"
else
    exp_suffix="exp${experiment_mode}"
fi

# Replace dots with underscores in static_sparsity for filename compatibility
static_sparsity_clean=$(echo $static_sparsity | sed 's/\./_/g')

# Create suffix similar to niah_test.sh format
suffix=sp${sparse_prefill_mode}_${precision}_${kv_quant_granularity}_ss${static_sparsity_clean}_sd${sparse_decode_mode}_budget${dynamic_sparse_token_budget}_int${selector_update_interval}_${exp_suffix}

if [ "$sparse_prefill_mode" == "1" ]; then
    suffix=${suffix}_context_S${ctx_sink_token}L${ctx_local_token}
elif [ "$sparse_prefill_mode" = "0" ]; then
    suffix=${suffix}
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']. Now "$sparse_prefill_mode. 
fi

echo "suffix: $suffix"

longbench_args="--base_model $base_model \
                --quant_model $ckpt_name \
                --model_path $model_path \
                --task $task \
                --sparse_prefill_mode $sparse_prefill_mode \
                --model_name_suffix $suffix"
                


lserve_args="--ifb-mode \
             --precision $precision \
             --quant-path $model_path \
             --group-size -1 \
             --max-num-batched-tokens 4195000 \
             --max-num-seqs 1 \
             --omit-prompt \
             --kv-quant-granularity $kv_quant_granularity \
             --chunk-prefill-size 32000 \
             --multiblock-switch 1024000 \
             --static-sparsity $static_sparsity \
             --sparse-decode-mode $sparse_decode_mode \
             --ctx-sink-token $ctx_sink_token \
             --ctx-local-token $ctx_local_token \
             --dec-sink-token $dec_sink_token \
             --dec-local-token $dec_local_token \
             --sub-chunk-per-block $sub_chunk_per_block \
             --dynamic-sparse-token-budget $dynamic_sparse_token_budget \
             --selector-update-interval $selector_update_interval"

# Only add attention pattern path if static_sparsity is not 0
if [ "$static_sparsity" != "0" ] && [ "$static_sparsity" != "0.0" ]; then
    lserve_args="$lserve_args --static-sparse-attn-load-dir $attn_pattern_path"
fi


if [ "$sparse_prefill_mode" == "1" ]; then
    CUDA_VISIBLE_DEVICES=${device} python -u pred.py $longbench_args $lserve_args --sparse-context-mode
elif [ "$sparse_prefill_mode" = "0" ]; then
    CUDA_VISIBLE_DEVICES=${device} python -u pred.py $longbench_args $lserve_args
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']."
fi 2>&1 | tee logs/eval_${model_name}_${task}_${suffix}.log

# GPU memory cleanup
echo "Cleaning up GPU memory..."
python -c "import torch; import gc; torch.cuda.empty_cache(); torch.cuda.synchronize(); gc.collect(); print('GPU memory cleanup completed')" 2>/dev/null || true