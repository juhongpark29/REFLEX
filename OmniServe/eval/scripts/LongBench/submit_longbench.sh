base_model="Llama-3-8B-Instruct-Gradient-4194k"
attn_path=./attn_patterns/Llama-3-8B-Instruct-Gradient-4194k
# model_path=./models/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
model_path=./models/Llama-3-8B-Instruct-Gradient-4194k-w8a8kv4-per-channel

if [ ! -d "$model_path" ]; then
    mkdir -p models
    cd models
    git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-Gradient-1048k-w8a8-per-channel-kv8-per-tensor
    cd ..
fi

cd eval/LongBench
if [ ! -d "$model_path" ]; then
    ln -s ../../models .
fi
if [ ! -d "$attn_path" ]; then
    ln -s ../../attn_patterns .
fi
cd ../..


# NOTE: For pure dense baseline, please set static_sparsity=0.0, sparse_prefill_mode=0, and sparse_decode_mode=0

# task_list=("2wikimqa" "dureader" "hotpotqa" "multi_news" "qasper" "qmsum" "samsum" "triviaqa")
task_list=("gov_report") #second cycle: english only
static_sparsity=0.3

# sparse_prefill_mode=1
sparse_prefill_mode=1
precision="w8a8kv4"
kv_quant_granularity=fine_grained
#sparse_decode_mode=0
sparse_decode_mode=1
dynamic_attn_budget=4096
dynamic_select_interval=4
sub_chunk_per_block=4

device=2

export CUDA_VISIBLE_DEVICES=$device
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="/workspace/OmniServe(backup):$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export NUM_RETRIEVAL_GPU_PAGE_BLOCKS=1500
export NUM_STREAMING_GPU_PAGE_BLOCKS=10
export TOKENS_PER_BLOCK=64
export EXPERIMENT_MODE=4
export ENABLE_PAGE_SELECTION_LOG=False
export PAGE_SELECTION_LOG_DIR="./my_page_logs"

ckpt_name=$(basename "$model_path")

for task in ${task_list[@]}; do
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=5000 \
    NUM_STREAMING_GPU_PAGE_BLOCKS=500 \
    bash eval/scripts/LongBench/longbench.sh \
    $base_model $model_path $attn_path \
    $task \
    $static_sparsity $sparse_prefill_mode \
    $precision $kv_quant_granularity \
    $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block \
    $device &

    device=$((device + 1))
done

wait

# Calculate current experiment suffix to filter evaluation files
experiment_mode=${EXPERIMENT_MODE}
multiple_value=${MULTIPLE_VALUE}
multiple_percentage=${MULTIPLE_PERCENTAGE}

if [ "$experiment_mode" == "4" ]; then
    exp_suffix="exp${experiment_mode}_G${multiple_value}${multiple_percentage}"
else
    exp_suffix="exp${experiment_mode}"
fi

static_sparsity_clean=$(echo $static_sparsity | sed 's/\./_/g')
suffix_pattern="ss${static_sparsity_clean}_sd${sparse_decode_mode}_budget${dynamic_attn_budget}_int${dynamic_select_interval}_${exp_suffix}"

echo "Evaluating files matching pattern: $suffix_pattern"

cd eval/LongBench
python -u eval.py --model $ckpt_name --file_pattern "$suffix_pattern"
cd ../..