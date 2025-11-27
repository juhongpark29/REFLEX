base_model="Llama-3-8B-Instruct-Gradient-1048k"
attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-1048k
model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor


# Download model if it does not exist
if [ ! -d "$model_path" ]; then
    mkdir -p models
    cd models
    git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
    cd ..
fi
# Ensure eval/needle has symlinks to models and attn_patterns
cd eval/needle
if [ ! -d "$model_path" ]; then
    ln -s ../../models .
fi
if [ ! -d "$attn_path" ]; then
    ln -s ../../attn_patterns .
fi
cd ../..

model_provider=LServe

##############################
# Needle-in-a-Haystack setup
##############################
# Document length sweep (start / end / #points)
s_len=1000    # start context length
e_len=256000  # end context length
n_col=12      # number of evaluation points
    
# LServe params
# For pure dense baseline, please set static_sparsity=0.0, sparse_prefill_mode=0, and sparse_decode_mode=0
static_sparsity=0.0

sparse_prefill_mode=0
precision="w8a8kv8"
kv_quant_granularity=per_tensor

sparse_decode_mode=0

dynamic_attn_budget=4096
dynamic_select_interval=4
sub_chunk_per_block=4

device=1

##############################
# Environment variables
##############################
# These settings override those in niah_test.sh
export CUDA_VISIBLE_DEVICES=$device
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="/workspace/OmniServe_release:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=3000
export NUM_STREAMING_GPU_PAGE_BLOCKS=10
export TOKENS_PER_BLOCK=64
export EXPERIMENT_MODE=4
export RULE_BASED_MODE=W4R8
export MULTIPLE_PERCENTAGE=90
export FILL_REMAINING=true
export ENABLE_PAGE_SELECTION_LOG=False
export PAGE_SELECTION_LOG_DIR="./my_page_logs"

bash eval/scripts/needle/niah_test.sh \
    $model_path $s_len $e_len $n_col \
    $model_provider $attn_path \
    $static_sparsity $sparse_prefill_mode \
    $precision $kv_quant_granularity \
    $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block