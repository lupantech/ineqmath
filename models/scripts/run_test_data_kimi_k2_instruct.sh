# https://platform.openai.com/docs/models

# cd /root/Projects/inequality_dev_2025/models/baselines

# Define array of LLM engines to test
ENGINES=(
    "together-moonshotai/Kimi-K2-Instruct"
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
    --max_workers 3 \
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
