import json
import os
import argparse

def combine_files(bound_data_file, relation_data_file, output_file):
    # File paths
    
    # Check if files exist
    if not os.path.exists(bound_data_file):
        print(f"Error: {bound_data_file} not found")
        return
    if not os.path.exists(relation_data_file):
        print(f"Error: {relation_data_file} not found")
        return
    
    print("Reading bound data file...")
    with open(bound_data_file, 'r', encoding='utf-8') as f:
        bound_data = json.load(f)
    
    print("Reading relation data file...")
    with open(relation_data_file, 'r', encoding='utf-8') as f:
        relation_data = json.load(f)
    
    # Combine all problems
    all_problems = []
    
    # Add problems from bound data
    if isinstance(bound_data, list):
        all_problems.extend(bound_data)
    else:
        print("Warning: bound_data is not a list, assuming it's a single problem")
        all_problems.append(bound_data)
    
    # Add problems from relation data
    if isinstance(relation_data, list):
        all_problems.extend(relation_data)
    else:
        print("Warning: relation_data is not a list, assuming it's a single problem")
        all_problems.append(relation_data)
    
    print(f"Total problems found: {len(all_problems)}")
    
    # Renumber data_id starting from 1252 and add required fields
    for i, problem in enumerate(all_problems):
        problem['data_id'] = str(i)  # Convert to string to match likely format
        problem['data_split'] = 'train'
        problem['theorems'] = {}
    
    # Sort by int(data_id)
    all_problems.sort(key=lambda x: int(x['data_id']))
    
    print(f"Writing {len(all_problems)} problems to {output_file}...")
    
    # Write combined data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_problems, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully created {output_file}")
    print(f"Data IDs range from {all_problems[0]['data_id']} to {all_problems[-1]['data_id']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine bound and relation data files')
    parser.add_argument('--bound_data_file', default="./raw/reformulated_bound_data.json", help='Path to the bound data JSON file')
    parser.add_argument('--relation_data_file', default="./raw/reformulated_relation_data.json", help='Path to the relation data JSON file')
    parser.add_argument('--output_file', default="./raw/combined_training_data.json", help='Path to the output combined JSON file')
    
    args = parser.parse_args()
    combine_files(args.bound_data_file, args.relation_data_file, args.output_file)
