# ==============================================
# This scripts is for vllm only.
# Define the models and max tokens below. 
# ==============================================

ENGINES=(
    "Qwen/Qwen3-4B"
)

TOKENS=10000


# ==============================================
# The following content does not need to be changed.
# ==============================================

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

    LABEL=${LOCAL_MODEL_NAME}_tokens_${TOKENS}

    python solve.py \
    --test_data_path ../../data/json/test.json\
    --output_path ../../results/models_results_test_data/\
    --llm_engine_name $ENGINE_NAME \
    --run_label $LABEL \
    --use_cache \
    --max_workers 5 \
    --max_tokens $TOKENS \
    --test_num -1

    python generate_results.py \
    --result_dir ../../results/models_results_test_data/\
    --run_label $LABEL \
    --use_cache \
    --max_workers 32

    echo "Completed tests for HF model: $HF_MODEL_NAME"
    echo "----------------------------------------"
done
