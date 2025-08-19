############ Configurations ############
INPUT_FILE="./data/numina_sampled_example.json"
LABEL="numina_sampled_example"
LLM_ENGINE="gpt-4.1"
MAX_WORKERS=8

############ Run the tasks ############
echo "Starting Task Reformulation Pipeline with $LLM_ENGINE..."

echo "Step 1/4: Running relation reformulation..."
python relation_reformulation.py \
    --llm_engine_name $LLM_ENGINE \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --input_file $INPUT_FILE \
    --output_file "./reformulated_data/$LABEL/raw/reformulated_relation_data.json" \
    --individual_files_dir "./reformulated_data/$LABEL/raw/relation_problems"

echo "Step 2/4: Running bound reformulation..."
python bound_reformulation.py \
    --llm_engine_name $LLM_ENGINE \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --input_file $INPUT_FILE \
    --output_file "./reformulated_data/$LABEL/raw/reformulated_bound_data.json" \
    --individual_files_dir "./reformulated_data/$LABEL/raw/bound_problems"

echo "Step 3/4: Combining reformulated data..."
python combine_files.py \
    --bound_data_file "./reformulated_data/$LABEL/raw/reformulated_bound_data.json" \
    --relation_data_file "./reformulated_data/$LABEL/raw/reformulated_relation_data.json" \
    --output_file "./reformulated_data/$LABEL/raw/combined_training_data.json"

echo "Step 4/4: Running solution reformulation..."
python solution_reformulation.py \
    --llm_engine_name $LLM_ENGINE \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --input_file "./reformulated_data/$LABEL/raw/combined_training_data.json" \
    --output_file "./reformulated_data/$LABEL/reformulated_data.json" \
    --individual_files_dir "./reformulated_data/$LABEL/raw/solution_reformulation"

echo "Pipeline completed! Output: ./reformulated_data/$LABEL/reformulated_data.json"
