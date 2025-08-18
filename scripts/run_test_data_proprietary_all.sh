############## Configurations ###############
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
SPLIT=test
OUTPUT_PATH="../../results/models_results_${SPLIT}_data/"
MAX_WORKERS=16

############## Run the model ###############
# Change working directory to utils
cd ../models/utils

# Loop through each engine
for LLM in "${ENGINES[@]}"; do
    echo "Running tests for engine: $LLM"

    LABEL=${LLM}_tokens_${TOKENS}

    python solve.py \
    --data_path ../../data/json/${SPLIT}.json\
    --split $SPLIT\
    --output_path $OUTPUT_PATH\
    --llm_engine_name $LLM \
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --max_tokens $TOKENS \
    --test_num -1

    python generate_results.py \
    --result_dir $OUTPUT_PATH\
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS

    echo "Completed tests for engine: $LLM"
    echo "----------------------------------------"
done
