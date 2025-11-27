cd eval/needle
mkdir -p logs img

model_path=$1
s_len=$2
e_len=$3
num_len=$4
model_provider=$5
attn_path=$6
static_sparsity=$7
sparse_prefill_mode=$8
precision=$9
kv_quant_granularity=${10}
sparse_decode_mode=${11}
dynamic_attn_budget=${12}
dynamic_select_interval=${13}
sub_chunk_per_block=${14}

ctx_sink_token=128
ctx_local_token=8192
dec_sink_token=128
dec_local_token=256

# NOTE: Naming convention for suffix:
#   sparse_prefill_mode=1        -> sp1
#   static_sparsity=0.5          -> ss0_5
#   sparse_decode_mode=1         -> sd1
#   dynamic_attn_budget=4096     -> budget4096
#   dynamic_select_interval=4    -> int4
#   page_size=128, exp=1         -> pg128_exp1

########################################
# Experiment / selection configuration
########################################
# Export defaults so Python can read them via environment variables

# Page size (tokens per block)
export TOKENS_PER_BLOCK=${TOKENS_PER_BLOCK:-64}  # Default to 64 if not set

# EXPERIMENT_MODE:
#   0 = Modulo-based selection
#         select pages where (page_idx % MULTIPLE_VALUE == MODULO_VALUE)
#   1 = Rule-based selection
#         apply predefined WnRm patterns (e.g., W4R8)
export EXPERIMENT_MODE=${EXPERIMENT_MODE:-0}
# Parameters for modulo-based selection (Mode 0)      
export MULTIPLE_VALUE=${MULTIPLE_VALUE:-10}        # divisor
export MULTIPLE_PERCENTAGE=${MULTIPLE_PERCENTAGE:-90}  # for high score pages (100% = only multiple pages, <100% = mixed with high-score pages)
# Parameters for rule-based selection (Mode 1)
# Supported patterns: W4R8, W2R4, W2R8, W1R2, W1R4, W1R8
export RULE_BASED_MODE=${RULE_BASED_MODE:-W4R8}

# Debug: print current selection config
echo "[DEBUG] MULTIPLE_VALUE: $MULTIPLE_VALUE"
echo "[DEBUG] MULTIPLE_PERCENTAGE: $MULTIPLE_PERCENTAGE"
echo "[DEBUG] RULE_BASED_MODE: $RULE_BASED_MODE"

# Copy exported values into local variables for filename suffix construction
page_size=$TOKENS_PER_BLOCK
experiment_mode=$EXPERIMENT_MODE
multiple_value=$MULTIPLE_VALUE
multiple_percentage=$MULTIPLE_PERCENTAGE
rule_based_mode=$RULE_BASED_MODE

# Defensive cleanup (in case values include "KEY=VALUE" format)
experiment_mode=$(echo "$experiment_mode" | sed 's/.*=//g')  
page_size=$(echo "$page_size" | sed 's/.*=//g')
multiple_value=$(echo "$multiple_value" | sed 's/.*=//g')
multiple_percentage=$(echo "$multiple_percentage" | sed 's/.*=//g')
rule_based_mode=$(echo "$rule_based_mode" | sed 's/.*=//g')

# Build experiment suffix based on mode and parameters
if [ "$experiment_mode" == "0" ]; then
    percentage="$multiple_percentage"
    exp_suffix="exp${experiment_mode}_G${multiple_value}${percentage}"
elif [ "$experiment_mode" == "1" ]; then
    percentage="$multiple_percentage"
    exp_suffix="exp${experiment_mode}_${rule_based_mode}${percentage}"
else
    exp_suffix="exp${experiment_mode}"
fi

static_sparsity_clean=$(echo $static_sparsity | sed 's/\./_/g')

if [ "$EXPERIMENT_MODE" = "5" ] && [ -n "$EXP5_EMULATION_MODE" ]; then
    suffix=${exp_suffix}_emulation${EXP5_EMULATION_MODE}_sp${sparse_prefill_mode}_${precision}_${kv_quant_granularity}_ss${static_sparsity_clean}_sd${sparse_decode_mode}_budget${dynamic_attn_budget}_int${dynamic_select_interval}_pg${page_size}
else
    suffix=sp${sparse_prefill_mode}_${precision}_${kv_quant_granularity}_ss${static_sparsity_clean}_sd${sparse_decode_mode}_budget${dynamic_attn_budget}_int${dynamic_select_interval}_pg${page_size}_${exp_suffix}
fi

if [ "$sparse_prefill_mode" == "1" ]; then
    suffix=${suffix}_context_S${ctx_sink_token}L${ctx_local_token}
elif [ "$sparse_prefill_mode" = "0" ]; then
    suffix=${suffix}
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']. Now "$sparse_prefill_mode. 
fi

echo "suffix: $suffix"

model_name=$(basename "$model_path")

common_args="--s_len $s_len \
    --e_len $e_len \
    --context_lengths_num_intervals $num_len \
    --model_provider $model_provider \
    --model_name_suffix $suffix \
    --method $method \
    --model_path $model_path \
    --ifb-mode \
    --precision $precision \
    --quant-path $model_path \
    --group-size -1 \
    --max-num-batched-tokens 4195000 \
    --max-num-seqs 1 \
    --omit-prompt \
    --kv-quant-granularity $kv_quant_granularity \
    --chunk-prefill-size 32000 \
    --multiblock-switch 32000 \
    --static-sparse-attn-load-dir $attn_path \
    --static-sparsity $static_sparsity \
    --sparse-decode-mode $sparse_decode_mode \
    --ctx-sink-token $ctx_sink_token \
    --ctx-local-token $ctx_local_token \
    --dec-sink-token $dec_sink_token \
    --dec-local-token $dec_local_token \
    --sub-chunk-per-block $sub_chunk_per_block \
    --dynamic-sparse-token-budget $dynamic_attn_budget \
    --selector-update-interval $dynamic_select_interval \
    --tensor-parallel-size 2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$sparse_prefill_mode" == "1" ]; then
    python -u needle_in_haystack.py $common_args --sparse-context-mode
elif [ "$sparse_prefill_mode" = "0" ]; then
    python -u needle_in_haystack.py $common_args
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']. Now "$sparse_prefill_mode. 
fi 2>&1 | tee logs/eval_${model_name}_${suffix}.log

python visualize.py \
    --folder_path "results/${model_name}_${suffix}/" \
    --model_name "${model_name} ${suffix}" \
    --pretrained_len 256000

