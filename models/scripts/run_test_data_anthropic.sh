############## Configurations ###############
ENGINES=(
     "claude-opus-4-20250514"
     "claude-sonnet-4-20250514"
     "claude-3-7-sonnet-20250219"
     "claude-3-5-sonnet-20240620"
     "claude-3-5-sonnet-20241022"
     "claude-3-5-haiku-20241022"
     "claude-3-opus-20240229"
     "claude-3-haiku-20240307"
     # You can add more models here
)
TOKENS=10000
SPLIT=test
OUTPUT_PATH="../../results/models_results_${SPLIT}_data/"
MAX_WORKERS=16

############## Run the model ###############
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
