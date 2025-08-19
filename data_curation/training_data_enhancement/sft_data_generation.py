import json
import re
import sys
import os
from typing import Optional
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm

# Add path to import the engines
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

from pydantic import BaseModel

# Define the hint prompts
BOUND_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is $C=X$', where X is your calculated numerical bound value. Example: 'The answer is $C=1$'."
RELATION_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is (Letter) Symbol', where Letter is one of the given options. Example: 'The answer is (A) $\\leq$'."

# Define structured output schema (keeping for reference, but won't use with engine)
class EnhancedSolution(BaseModel):
    enhanced_solution: str

def get_enhancement_prompt(problem_type, problem, current_solution, answer):
    """
    Create enhancement prompt based on problem type
    """
    base_prompt = """You are an expert mathematician tasked with enhancing a mathematical solution to make it more detailed, rigorous, and logically sound.

Given a problem and its current solution, please provide an enhanced version that includes:
1. More detailed step-by-step reasoning
2. Clear logical connections between steps
3. Rigorous mathematical justifications

IMPORTANT: Do not change how the problem is solved in the original response! Keep the same approach, methodology, and solution strategy. Only enhance the clarity, detail, and rigor of the explanations while maintaining the original solving approach.

The enhanced solution should maintain the same final answer but provide much more comprehensive reasoning."""

    if problem_type == "bound":
        specific_prompt = f"""
Problem Type: Bound (finding numerical bounds)
Required Answer Format: "The answer is $C=X$" where X is the numerical value.

Original Problem: {problem}
Current Solution: {current_solution}
Expected Answer: {answer}

Please provide an enhanced_solution that is more detailed and logically solid while maintaining mathematical rigor. Make sure to end with the exact answer format required."""
    
    elif problem_type == "relation":
        specific_prompt = f"""
Problem Type: Relation (determining mathematical relationships)
Required Answer Format: "The answer is (Letter) Symbol" where Letter corresponds to the given options.

Original Problem: {problem}
Current Solution: {current_solution}
Expected Answer: {answer}

Please provide an enhanced_solution that is more detailed and logically solid while maintaining mathematical rigor. Make sure to end with the exact answer format required."""
    
    else:
        specific_prompt = f"""
Original Problem: {problem}
Current Solution: {current_solution}
Expected Answer: {answer}

Please provide an enhanced_solution that is more detailed and logically solid while maintaining mathematical rigor."""

    return base_prompt + specific_prompt

def get_enhanced_solution(llm_engine, problem_type, problem, current_solution, answer, max_retries=3):
    """
    Use LLM engine to get enhanced solution
    """
    prompt = get_enhancement_prompt(problem_type, problem, current_solution, answer)
    
    for attempt in range(max_retries):
        try:
            # Use the engine factory approach like in one_shot_solve.py
            response = llm_engine(prompt, max_tokens=10000, response_format=EnhancedSolution)
            
            if response.enhanced_solution:
                return response.enhanced_solution
            else:
                print(f"Attempt {attempt + 1}: Got empty response")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to enhance solution after {max_retries} attempts. Using original solution.")
                return f"{current_solution} Therefore, the answer is {answer}."
    
    return f"{current_solution} Therefore, the answer is {answer}."

def clean_answer(answer):
    """
    Clean the answer by removing double dollar signs from both sides
    For example: $$C = 3$$ becomes $C = 3$
    """
    if isinstance(answer, str):
        # Remove double dollar signs from both sides
        answer = re.sub(r'^\$\$(.+)\$\$$', r'$\1$', answer.strip())
    return answer

def extract_solution(solution):
    """
    Extract solution - if it's a list, get the first element
    """
    if isinstance(solution, list):
        return solution[0] if solution else ""
    return solution

def get_hint_prompt(entry_type):
    """
    Get the appropriate hint prompt based on entry type
    """
    if entry_type == "bound":
        return BOUND_HINT_PROMPT
    elif entry_type == "relation":
        return RELATION_HINT_PROMPT
    else:
        return ""  # Default to no hint if type is unknown

def create_markdown_content(problem, current_solution, answer, enhanced_solution, entry_type):
    """
    Create markdown content for the processed entry
    """
    markdown_content = f"""# Problem Entry

## Problem Type
{entry_type}

## Problem
{problem}

## Current Solution
{current_solution}

## Expected Answer
{answer}

## Enhanced Solution
{enhanced_solution}
"""
    return markdown_content

def save_individual_files(entry_data, output_dir, problem, current_solution, answer, enhanced_solution, entry_type):
    """
    Save individual markdown and JSON files for each entry
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data_id from entry
    data_id = entry_data.get('data_id', entry_data.get('annot_id', f"entry_{hash(problem) % 10000}"))
    
    # Save markdown file
    markdown_content = create_markdown_content(problem, current_solution, answer, enhanced_solution, entry_type)
    md_path = os.path.join(output_dir, f"{data_id}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Save JSON file
    json_data = {
        "data_id": data_id,
        "type": entry_type,
        "problem": problem,
        "current_solution": current_solution,
        "answer": answer,
        "enhanced_solution": enhanced_solution,
        "original_entry": entry_data
    }
    json_path = os.path.join(output_dir, f"{data_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def process_entry(llm_engine, entry, individual_files_dir, entry_index):
    """
    Process a single entry for SFT data generation
    """
    # Extract required fields
    problem = entry.get('problem', '')
    solution = entry.get('solution', '')
    answer = entry.get('answer', '')
    entry_type = entry.get('type', '')
    
    # Skip if any required field is missing
    if not problem or not solution or not answer:
        return {
            'success': False,
            'reason': 'missing_fields',
            'entry_index': entry_index
        }
    
    # Process solution and answer according to requirements
    processed_solution = extract_solution(solution)
    processed_answer = clean_answer(answer)
    
    # Get data_id for checking existing files
    data_id = entry.get('data_id', entry.get('annot_id', f"entry_{hash(problem) % 10000}"))
    existing_json_path = os.path.join(individual_files_dir, f"{data_id}.json")
    
    # Check if individual JSON file already exists
    enhanced_solution = None
    skipped = False
    if os.path.exists(existing_json_path):
        try:
            with open(existing_json_path, 'r', encoding='utf-8') as existing_f:
                existing_data = json.load(existing_f)
                enhanced_solution = existing_data.get('enhanced_solution')
                if enhanced_solution:
                    skipped = True
                else:
                    print(f"Existing file found but no enhanced_solution field for entry {entry_index + 1} (ID: {data_id})")
        except Exception as e:
            print(f"Error reading existing file {existing_json_path}: {str(e)}")
    
    # Generate enhanced solution only if not found in existing file
    if not enhanced_solution:
        try:
            enhanced_solution = get_enhanced_solution(
                llm_engine,
                entry_type, 
                problem, 
                processed_solution, 
                processed_answer
            )
            
            # Save individual markdown and JSON files
            save_individual_files(
                entry, 
                individual_files_dir, 
                problem, 
                processed_solution, 
                processed_answer, 
                enhanced_solution, 
                entry_type
            )
        except Exception as e:
            return {
                'success': False,
                'reason': 'generation_failed',
                'error': str(e),
                'entry_index': entry_index,
                'data_id': data_id
            }
    
    # Get the appropriate hint prompt
    hint_prompt = get_hint_prompt(entry_type)
    
    # Create the user content with hint prompt + problem
    if hint_prompt:
        user_content = f"{hint_prompt}\n\nProblem: {problem}\n\nSolution:"
    else:
        user_content = problem
    
    # Create the fine-tuning format
    sft_entry = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant", 
                "content": enhanced_solution
            }
        ]
    }
    
    return {
        'success': True,
        'sft_entry': sft_entry,
        'skipped': skipped,
        'entry_index': entry_index,
        'data_id': data_id
    }

def combine_individual_json_files(individual_files_dir, output_combined_file):
    """
    Combine all individual JSON files into a single large JSON file.
    Each entry in the output will be the original_entry with enhanced_solution added as a new field.
    """
    combined_data = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(individual_files_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Warning: No JSON files found in {individual_files_dir}")
        return
    
    print(f"Combining {len(json_files)} JSON files into {output_combined_file}")
    
    for json_file in tqdm(json_files, desc="Combining JSON files"):
        json_path = os.path.join(individual_files_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # Extract original_entry and enhanced_solution
            original_entry = file_data.get('original_entry', {})
            enhanced_solution = file_data.get('enhanced_solution', '')
            
            # Create new entry with original_entry data plus enhanced_solution
            combined_entry = original_entry.copy()
            combined_entry['enhanced_solution'] = enhanced_solution
            
            combined_data.append(combined_entry)
            
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
    
    # Sort the combined data by data_id converted to integer
    def get_sort_key(entry):
        try:
            data_id = entry.get('data_id', entry.get('annot_id', '0'))
            return int(data_id)
        except (ValueError, TypeError):
            # If conversion fails, put at the end with a large number
            return float('inf')
    
    combined_data.sort(key=get_sort_key)
    
    # Save the combined data
    try:
        with open(output_combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully combined {len(combined_data)} entries into {output_combined_file} (sorted by data_id)")
    except Exception as e:
        print(f"Error saving combined file: {str(e)}")

def create_sft_data(llm_engine_name="gpt-4o-mini", use_cache=False, max_workers=1, input_file="../../../data/final_data_huggingface_250721/train.json", output_jsonl_file="./sft_data.jsonl", individual_files_dir="./raw", output_json_file=None):
    # Import and create LLM engine like in one_shot_solve.py
    from models.engines.factory import create_llm_engine
    llm_engine = create_llm_engine(llm_engine_name, use_cache)
    
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return
    # Ensure the directory for output_jsonl_file exists
    output_dir = os.path.dirname(output_jsonl_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # If the output_jsonl_file doesn't exist, create it
    if not os.path.exists(output_jsonl_file):
        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            pass
    # If the individual_files_dir doesn't exist, create it
    if not os.path.exists(individual_files_dir):
        os.makedirs(individual_files_dir, exist_ok=True)
    
    # Process the data and create JSONL entries
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    
    with open(output_jsonl_file, 'w', encoding='utf-8') as f:
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_entry, llm_engine, entry, individual_files_dir, i)
                    for i, entry in enumerate(data)
                ]
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing entries"):
                    result = future.result()
                    
                    if result['success']:
                        sft_entry = result['sft_entry']
                        f.write(json.dumps(sft_entry, ensure_ascii=False) + '\n')
                        processed_count += 1
                        if result['skipped']:
                            skipped_count += 1
                            print(f"Using existing enhanced solution for entry {result['entry_index'] + 1} (ID: {result['data_id']})")
                        else:
                            print(f"Generated enhanced solution for entry {result['entry_index'] + 1} (ID: {result['data_id']})")
                    else:
                        print(f"Failed to process entry {result['entry_index'] + 1}: {result['reason']}")
                        failed_count += 1
        else:
            for i, entry in enumerate(tqdm(data, desc="Processing entries")):
                result = process_entry(llm_engine, entry, individual_files_dir, i)
                
                if result['success']:
                    sft_entry = result['sft_entry']
                    f.write(json.dumps(sft_entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if result['skipped']:
                        skipped_count += 1
                        print(f"Using existing enhanced solution for entry {result['entry_index'] + 1} (ID: {result['data_id']})")
                    else:
                        print(f"Generated enhanced solution for entry {result['entry_index'] + 1} (ID: {result['data_id']})")
                else:
                    print(f"Failed to process entry {result['entry_index'] + 1}: {result['reason']}")
                    failed_count += 1
    
    print(f"Successfully processed {processed_count} entries")
    print(f"Skipped {skipped_count} entries (using existing enhanced solutions)")
    print(f"Failed to process {failed_count} entries")
    print(f"Output saved to {output_jsonl_file}")
    print(f"Individual files saved to {individual_files_dir}/")
    
    # Combine individual JSON files into a single large JSON file if requested
    if output_json_file:
        print("\nCombining individual JSON files...")
        combine_individual_json_files(individual_files_dir, output_json_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4.1-mini", help="LLM engine to use for enhancement")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Use cache for LLM calls")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--input_file", type=str, default="../../../data/final_data_huggingface_250721/train.json", help="Path to input JSON file")
    parser.add_argument("--output_jsonl_file", type=str, default="./sft_data.jsonl", help="Path to output JSONL file")
    parser.add_argument("--individual_files_dir", type=str, default="./raw", help="Directory to save individual markdown and JSON files")
    parser.add_argument("--output_json_file", type=str, default=None, help="Path to output combined JSON file (optional)")
    args = parser.parse_args()
    
    create_sft_data(args.llm_engine_name, args.use_cache, args.max_workers, args.input_file, args.output_jsonl_file, args.individual_files_dir, args.output_json_file)
