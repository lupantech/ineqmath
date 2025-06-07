# https://platform.openai.com/docs/models

# cd /root/Projects/inequality_dev_2025/models/baselines

# Define array of LLM engines to test
ENGINES=(
    "gemini-2.0-flash"
    "gemini-2.0-flash-lite"
    "gpt-4o-2024-08-06"
    "gpt-4o-mini-2024-07-18"
    "gpt-4.1-2025-04-14"
    "grok-3-beta"
    "claude-3-7-sonnet-20250219"
    "gemini-2.5-flash-preview-04-17"
    "gemini-2.5-pro-preview-05-06"
    "grok-3-mini-beta"
    "o1-2024-12-17"
    "o3-2025-04-16"
    "o3-mini-2025-01-31"
    "o4-mini-2025-04-16"   
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
