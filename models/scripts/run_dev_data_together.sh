############## Configurations ###############
ENGINES=(
    "together-moonshotai/Kimi-K2-Instruct"
    "together-meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "together-Qwen/QwQ-32B"
    "together-Qwen/Qwen2-VL-72B-Instruct"
    # You can add more models here
)
TOKENS=10000
SPLIT=dev
OUTPUT_PATH="../../results/models_results_${SPLIT}_data/"
MAX_WORKERS=16

############## Run the model ###############
# Change working directory to utils
cd ../utils

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

    python compute_score.py \
    --result_dir $OUTPUT_PATH \
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS

    echo "Completed tests for engine: $LLM"
    echo "----------------------------------------"
done
