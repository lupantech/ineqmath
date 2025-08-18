import os
import json
import argparse
import re
from typing import List, Dict, Any

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="Directory containing the results")
    parser.add_argument("--run_label", type=str, required=True, help="Label of the run to process")
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=1)
    return parser.parse_args()

def natural_sort_key(s: str) -> List:
    """Helper function to sort strings with numbers naturally."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def read_json_files(directory: str) -> List[Dict[str, Any]]:
    """Read all JSON files from the specified directory and return them as a list."""
    results = []
    # Get all json files and sort them naturally
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files.sort(key=natural_sort_key)
    
    for filename in json_files:
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Reorder the dictionary to put data_id first
                if 'data_id' in data:
                    reordered_data = {'data_id': data['data_id']}
                    for key, value in data.items():
                        if key != 'data_id':
                            reordered_data[key] = value
                    results.append(reordered_data)
                else:
                    results.append(data)
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
    return results

def main():
    args = parse_arguments()
    
    # Construct the path to the raw results directory
    raw_dir = os.path.join(args.result_dir, args.run_label, "raw")
    
    if not os.path.exists(raw_dir):
        print(f"Directory not found: {raw_dir}")
        return
    
    # Read all JSON files
    print(f"Reading JSON files from {raw_dir}...")
    results = read_json_files(raw_dir)
    
    # Save the combined results
    output_file = os.path.join(args.result_dir, args.run_label, "results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Successfully combined {len(results)} results into {output_file}")

if __name__ == "__main__":
    main() 