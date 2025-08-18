############## Configurations ###############
ENGINES=(
    "grok-3-mini-beta"
    "o3-mini-2025-01-31"
    # You can add more models here 
    # For vllm, you need to use another script run_test_data_vllm.sh
)
TOKENS=10000
SPLIT=train
OUTPUT_PATH="../../results/training_theorems_as_hints_results_${SPLIT}_data/"
MAX_WORKERS=16

############## Run the model ###############
# Change working directory to utils
cd ../../models/utils

# Loop through each engine
for LLM in "${ENGINES[@]}"; do
    echo "Running tests for engine: $LLM"

    LABEL=${LLM}_tokens_${TOKENS}

    python solve_theorem_as_hints.py \
    --data_path ../../data/json/training_data_sampled_200.json\
    --split $SPLIT\
    --output_path $OUTPUT_PATH\
    --llm_engine_name $LLM \
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --max_tokens $TOKENS \
    --test_num -1 \

  

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
