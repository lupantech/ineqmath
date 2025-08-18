############## Configurations ###############
ENGINES=(
    "gpt-4o-mini"
    # You can add more models here
)
TOKENS=10000
SPLIT=dev
OUTPUT_PATH="../../results/models_results_${SPLIT}_data/"
MAX_WORKERS=16
SHOT_NUM=0 # 1,2,3

############## Run the model ###############
# Change working directory to utils
cd ../utils

# Loop through each engine
for LLM in "${ENGINES[@]}"; do
    echo "Running tests for engine: $LLM"

    LABEL=${LLM}_tokens_${TOKENS}_shot_num_${SHOT_NUM}

    python solve_few_shot.py \
    --data_path ../../data/json/${SPLIT}.json\
    --split $SPLIT\
    --output_path $OUTPUT_PATH\
    --llm_engine_name $LLM \
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --max_tokens $TOKENS \
    --shot_num $SHOT_NUM \
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
