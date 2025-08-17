# ==============================================
# This scripts is for vllm only.
# Heres are the configurations.
# ==============================================
ENGINES=(
     "Qwen/Qwen3-4B"
     # You can add more vllm models here
)
TOKENS=10000
SPLIT=dev
OUTPUT_PATH="../../results/frequent_theorems_as_hints_results_${SPLIT}_data/"
MAX_WORKERS=16
THEOREM_NUM=3 # 1,2,3

# ==============================================
# The following content does not need to be changed.
# ==============================================

# Change working directory to utils
cd ../utils

echo "----------------------------------------"
echo "Starting model download process..."
echo "----------------------------------------"

# Download models from ENGINES array
for HF_MODEL_NAME in "${ENGINES[@]}"; do
    # Extract the model name after '/' for local directory name
    LOCAL_MODEL_NAME=$(echo "$HF_MODEL_NAME" | sed 's/.*\///')
    LOCAL_DIR="local_models/${LOCAL_MODEL_NAME}"
    
    echo "Downloading model: ${HF_MODEL_NAME}"
    echo "Local directory: ${LOCAL_DIR}"
    echo "-----------------------------------------------"
    
    # Download the model from HuggingFace using correct arguments
    python download_local_models.py \
        --repo_id "${HF_MODEL_NAME}" \
        --local_dir "${LOCAL_DIR}"
    
    echo "Completed download for: ${HF_MODEL_NAME}"
    echo "----------------------------------------"
done

echo "All models downloaded successfully!"
echo "----------------------------------------"

# Loop through each HuggingFace model name
for HF_MODEL_NAME in "${ENGINES[@]}"; do
    # Extract the model name after '/' for local engine name
    LOCAL_MODEL_NAME=$(echo "$HF_MODEL_NAME" | sed 's/.*\///')
    ENGINE_NAME="vllm-local_models/${LOCAL_MODEL_NAME}"
    
    echo "Running tests for HF model: $HF_MODEL_NAME"
    echo "Using engine name: $ENGINE_NAME"

    LABEL=${LOCAL_MODEL_NAME}_tokens_${TOKENS}_theorem_num_${THEOREM_NUM}

    python solve_theorem_as_hints.py \
    --data_path ../../data/json/${SPLIT}.json\
    --split $SPLIT\
    --output_path $OUTPUT_PATH\
    --llm_engine_name $ENGINE_NAME \
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

    echo "Completed tests for HF model: $HF_MODEL_NAME"
    echo "----------------------------------------"
done
