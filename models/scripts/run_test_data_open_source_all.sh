# https://platform.openai.com/docs/models

# cd /root/Projects/inequality_dev_2025/models/baselines

# Define array of LLM engines to test
ENGINES=(
    "meta-llama/together-Llama-4-Maverick-17B-128E-Instruct-FP8"
    "meta-llama/together-Llama-4-Scout-17B-16E-Instruct"
    "meta-llama/together-Meta-Llama-3.1-8B-Instruct-Turbo"
    "meta-llama/together-Llama-3.2-3B-Instruct-Turbo"
    "Qwen/together-Qwen2.5-Coder-32B-Instruct"
    "Qwen/together-Qwen2.5-7B-Instruct-Turbo"
    "Qwen/together-Qwen2.5-72B-Instruct-Turbo"
    "deepseek-ai/together-DeepSeek-R1-Distill-Llama-70B"
    "deepseek-ai/together-DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/together-Qwen3-235B-A22B-fp8-tput"
    "Qwen/together-QwQ-32B"
    "Qwen/together-QwQ-32B-Preview"   
)

TOKENS=10000

# Loop through each engine
for LLM in "${ENGINES[@]}"; do
    echo "Running tests for engine: $LLM"

    LABEL=${LLM}_tokens_${TOKENS}

    python solve.py \
    --test_data_path ../../data/json/test.json\
    --output_path ../../results/models_results_test_data/\
    --llm_engine_name $LLM \
    --run_label $LABEL \
    --use_cache \
    --max_workers 20 \
    --max_tokens $TOKENS \
    --test_num -1

    python generate_results.py \
    --result_dir ../../results/models_results_test_data/\
    --run_label $LABEL \
    --use_cache \
    --max_workers 32

    echo "Completed tests for engine: $LLM"
    echo "----------------------------------------"
done
