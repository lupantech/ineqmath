############## Configurations ###############
ENGINES=(
    "gpt-4o-mini"
    # You can add more models here 
    # For vllm, you need to use another script run_test_data_vllm.sh
)
TOKENS=10000
SPLIT=dev
OUTPUT_PATH="../../results/frequent_theorems_as_hints_results_${SPLIT}_data/"
MAX_WORKERS=16
THEOREM_NUM=3 # 1,2,3

############## Run the model ###############
# Change working directory to utils
cd ../utils

# Loop through each engine
for LLM in "${ENGINES[@]}"; do
    echo "Running tests for engine: $LLM"

    LABEL=${LLM}_tokens_${TOKENS}_theorem_num_${THEOREM_NUM}

    python solve_theorem_as_hints.py \
    --data_path ../../data/json/${SPLIT}.json\
    --split $SPLIT\
    --output_path $OUTPUT_PATH\
    --llm_engine_name $LLM \
    --run_label $LABEL \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --max_tokens $TOKENS \
    --test_num -1 \
    --theorem_num $THEOREM_NUM \
  

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
