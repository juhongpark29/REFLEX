model_name="Llama-3-8B-Instruct-Gradient-1048k"
attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-1048k
model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor

# Download model if not already present
if [ ! -d "$model_path" ]; then
  mkdir -p models
  cd models
  git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
  cd ..
fi

# Example alternative model (disabled)
# model_name="Mistral-7B-v0.1-QServe"
# attn_path=./attn_patterns/Mistral-7B-Instruct-v0.3
# model_path=./models/Mistral-7B-v0.1-QServe
# if [ ! -d "$model_path" ]; then
#   mkdir -p models && cd models
#   git clone https://huggingface.co/mit-han-lab/Mistral-7B-v0.1-QServe
#   cd ..
# fi

# NOTE: Precision configuration
# precision="w8a8kv8" # for llama3
precision="w8a8kv8" #for llama2
static_sparsity=0.3  # Static sparse attention ratio (alpha 0.0~1.0)

device=2
export CUDA_VISIBLE_DEVICES=$device

##############################
# Logging configuration
##############################
export ENABLE_PAGE_SELECTION_LOG=${ENABLE_PAGE_SELECTION_LOG:-false}  # Default: true (selected page)
export ENABLE_SCORE_LOG=${ENABLE_SCORE_LOG:-false}  # Default: true (page score)
export ENABLE_UNIFIED_LOG=${ENABLE_UNIFIED_LOG:-false}  # Default: true (intergrated log)
export PAGE_SELECTION_LOG_DIR=${PAGE_SELECTION_LOG_DIR:-"./my_page_logs"}  # Default: ./my_page_logs

echo "Page selection logging: $ENABLE_PAGE_SELECTION_LOG"
echo "Page score logging: $ENABLE_SCORE_LOG"
echo "Unified logging: $ENABLE_UNIFIED_LOG"
echo "Log directory: $PAGE_SELECTION_LOG_DIR"

##############################
# Experiment configuration
##############################
# EXPERIMENT_MODE:
#   0 = Modulo-based selection:
#         select pages satisfying (page_idx % MULTIPLE_VALUE == MODULO_VALUE)
#   1 = Rule-based selection:
#         apply predefined structural patterns (e.g., WnRm)

# Experiment configuration - set default values and export them

export EXPERIMENT_MODE=${EXPERIMENT_MODE:-4}
export TOKENS_PER_BLOCK=${TOKENS_PER_BLOCK:-64}

# Parameters for modulo-based selection (Mode 0)
export MULTIPLE_VALUE=${MULTIPLE_VALUE:-1}       # divisor
export MODULO_VALUE=${MODULO_VALUE:-0}           # selected residue: page_idx % MULTIPLE_VALUE == MODULO_VALUE
export MULTIPLE_PERCENTAGE=${MULTIPLE_PERCENTAGE:-70}  # for high score pages (100% = only multiple pages, <100% = mixed with high-score pages)
# Parameters for rule-based selection (Mode 1)
export RULE_BASED_MODE=${RULE_BASED_MODE:-W4R8}  # supported: W4R8, W2R4, W2R8, W1R2, W1R4, W1R8

##############################
# Run generation
##############################

NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=200 \
ENABLE_PAGE_SELECTION_LOG=$ENABLE_PAGE_SELECTION_LOG \
ENABLE_SCORE_LOG=$ENABLE_SCORE_LOG \
ENABLE_UNIFIED_LOG=$ENABLE_UNIFIED_LOG \
PAGE_SELECTION_LOG_DIR=$PAGE_SELECTION_LOG_DIR \
EXPERIMENT_MODE=$EXPERIMENT_MODE \
TOKENS_PER_BLOCK=$TOKENS_PER_BLOCK \
MULTIPLE_VALUE=$MULTIPLE_VALUE \
MODULO_VALUE=$MODULO_VALUE \
MULTIPLE_PERCENTAGE=$MULTIPLE_PERCENTAGE \
RULE_BASED_MODE=$RULE_BASED_MODE \
python lserve_e2e_generation.py \
  --model $model_path \
  --ifb-mode \
  --precision $precision \
  --quant-path $model_path \
  --group-size -1 \
  --max-num-batched-tokens 4195000 \
  --max-num-seqs 1 \
  --omit-prompt \
  --kv-quant-granularity "per_tensor" \
  --chunk-prefill-size 32000 \
  --multiblock-switch 2048 \
  --static-sparse-attn-load-dir $attn_path \
  --static-sparsity $static_sparsity \
  --sparse-context-mode \
  --sparse-decode-mode 1 \
  --ctx-sink-token 128 \
  --ctx-local-token 8192 \
  --dec-sink-token 128 \
  --dec-local-token 256 \
  --sub-chunk-per-block 4 \
  --dynamic-sparse-token-budget 4096 \
  --selector-update-interval 4
