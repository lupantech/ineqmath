#########Configuration#########
INPUT_FILE="../../data/json/train.json"
MAX_WORKERS=32
LABEL="training_data"

#########Run the script#########
echo "Running the script..."
python sft_data_generation.py \
    --llm_engine_name "o4-mini" \
    --use_cache \
    --max_workers $MAX_WORKERS \
    --input_file $INPUT_FILE \
    --output_jsonl_file ./enhanced_training_data/$LABEL/sft_data_$LABEL.jsonl \
    --individual_files_dir ./enhanced_training_data/$LABEL/raw \
    --output_json_file ./enhanced_training_data/$LABEL/enhanced_data_$LABEL.json
echo "Script completed."
echo "Output files are saved in ./enhanced_training_data/$LABEL/"


