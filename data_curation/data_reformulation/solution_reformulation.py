import json
import re
import sys
import os
from typing import Optional
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import glob

# Add path to import the engines
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables
load_dotenv()

from pydantic import BaseModel

# Define structured output schema
class EnhancedSolution(BaseModel):
    enhanced_solution: str

# read example file
with open("./reformulation_example.md", "r") as f:
    example = f.read()

def get_reformulation_prompt(problem_type, problem, current_solution, answer):
    """
    Create reformulation prompt based on problem type
    """
    if problem_type == "bound":
        prompt = f"""You are an expert mathematician. Given a bound problem and its current solution, please reformulate it by adding an explanation after the original solution about when the equality condition holds and what the maximum or minimum value of C is.

Please provide a reformulated solution that:
1. Keeps the original solution exactly as is
2. Adds an explanation after the original solution about when the equality is achieved/holds
3. Clearly states whether this gives the maximum or minimum value of C
4. Ends with "Therefore, the answer is {answer}."

Format your response as: [Original Solution] + [Your additional explanation about equality conditions and max/min] + "Therefore, the answer is {answer}."

Do not change the original solution approach or methodology - only add the equality condition explanation.

Please make the additional explanation as short as possible without losing clarity.

Here are some examples:
{example}

Please reformulate the following problem:
Original Problem: {problem}
Current Solution: {current_solution}
Expected Answer: {answer}
"""
    
    else:
        # This shouldn't be used for other types, but keeping for safety
        prompt = f"""
Original Problem: {problem}
Current Solution: {current_solution}
Expected Answer: {answer}

Please provide the solution as is."""

    return prompt

def get_reformulated_solution(llm_engine, problem_type, problem, current_solution, answer, max_retries=3):
    """
    Use LLM engine to get reformulated solution with equality conditions explanation
    """
    prompt = get_reformulation_prompt(problem_type, problem, current_solution, answer)
    
    for attempt in range(max_retries):
        try:
            response = llm_engine(prompt, max_tokens=10000, response_format=EnhancedSolution)
            
            if response.enhanced_solution:
                return response.enhanced_solution
            else:
                print(f"Attempt {attempt + 1}: Got empty response")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to reformulate solution after {max_retries} attempts. Using original solution.")
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

def process_single_solution(llm_engine, problem_type, problem, solution, answer):
    """
    Process a single solution string
    """
    processed_answer = clean_answer(answer)
    
    if problem_type == 'relation':
        # For relation problems, just add the answer at the end
        return f"{solution} Therefore, the answer is {processed_answer}."
    elif problem_type == 'bound':
        # For bound problems, reformulate by adding equality conditions explanation
        return get_reformulated_solution(
            llm_engine,
            problem_type, 
            problem, 
            solution, 
            processed_answer
        )
    else:
        # For other types, keep original solution
        return solution

def create_markdown_content(problem, current_solution, answer, reformulated_solution, entry_type):
    """
    Create markdown content for the processed entry
    """
    # Handle display of solutions that might be lists
    if isinstance(current_solution, list):
        current_solution_str = "\n\n".join([f"**Solution {i+1}:**\n{sol}" for i, sol in enumerate(current_solution)])
    else:
        current_solution_str = str(current_solution)
    
    if isinstance(reformulated_solution, list):
        reformulated_solution_str = "\n\n".join([f"**Solution {i+1}:**\n{sol}" for i, sol in enumerate(reformulated_solution)])
    else:
        reformulated_solution_str = str(reformulated_solution)
    
    markdown_content = f"""# Problem Entry

## Problem Type
{entry_type}

## Problem
{problem}

## Current Solution
{current_solution_str}

## Expected Answer
{answer}

## Reformulated Solution
{reformulated_solution_str}
"""
    return markdown_content

def save_individual_files(entry_data, output_dir, problem, current_solution, answer, reformulated_solution, entry_type):
    """
    Save individual markdown and JSON files for each entry
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data_id from entry
    data_id = entry_data.get('data_id', entry_data.get('annot_id', f"entry_{hash(problem) % 10000}"))
    
    # Save markdown file
    markdown_content = create_markdown_content(problem, current_solution, answer, reformulated_solution, entry_type)
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
        "reformulated_solution": reformulated_solution,
        "original_entry": entry_data
    }
    json_path = os.path.join(output_dir, f"{data_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def process_entry(llm_engine, entry, entry_index, individual_files_dir):
    """
    Process a single entry and modify only the solution field
    """
    # Create a copy of the entry to avoid modifying the original
    processed_entry = entry.copy()
    
    # Extract required fields
    problem = entry.get('problem', '')
    solution = entry.get('solution', '')
    answer = entry.get('answer', '')
    entry_type = entry.get('type', '')
    
    # Skip if any required field is missing
    if not problem or not solution or not answer:
        print(f"Skipping entry {entry_index + 1}: missing required fields")
        return processed_entry
    
    # Get data_id from entry to check if already processed
    data_id = entry.get('data_id', entry.get('annot_id', f"entry_{hash(problem) % 10000}"))
    
    # Check if this entry has already been processed
    json_path = os.path.join(individual_files_dir, f"{data_id}.json")
    md_path = os.path.join(individual_files_dir, f"{data_id}.md")
    
    if os.path.exists(json_path) and os.path.exists(md_path):
        print(f"Entry {entry_index + 1} (ID: {data_id}) already processed, skipping...")
        return processed_entry
    
    try:
        # Handle both string and list solutions
        if isinstance(solution, list):
            # Process each solution in the list
            new_solutions = []
            for i, single_solution in enumerate(solution):
                if single_solution:  # Skip empty solutions
                    reformulated = process_single_solution(llm_engine, entry_type, problem, single_solution, answer)
                    new_solutions.append(reformulated)
                else:
                    new_solutions.append(single_solution)
            new_solution = new_solutions
            current_solution = solution
            print(f"Processed {len(new_solutions)} solutions for entry {entry_index + 1} (ID: {data_id})")
        else:
            # Process single solution
            current_solution = solution
            new_solution = process_single_solution(llm_engine, entry_type, problem, solution, answer)
            print(f"Processed single solution for entry {entry_index + 1} (ID: {data_id})")
        
        # Update the solution field
        processed_entry['solution'] = new_solution
        
        # Save individual markdown and JSON files
        save_individual_files(
            entry, 
            individual_files_dir, 
            problem, 
            current_solution, 
            clean_answer(answer), 
            new_solution, 
            entry_type
        )
        
    except Exception as e:
        print(f"Error processing entry {entry_index + 1} (ID: {data_id}): {str(e)}")
        # Keep original solution if processing fails
    
    return processed_entry

def combine_json_files_from_raw(raw_dir, output_file):
    """
    Combine all JSON files from the raw directory to create the final dataset
    """
    print(f"Combining JSON files from {raw_dir}...")
    
    # Get all JSON files in the raw directory
    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {raw_dir}")
        return
    
    combined_data = []
    
    for json_file in tqdm(json_files, desc="Combining JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Extract the necessary fields to recreate the original format
            original_entry = data.get('original_entry', {})
            reformulated_solution = data.get('reformulated_solution', '')
            
            # Create new entry with reformulated solution
            new_entry = original_entry.copy()
            new_entry['solution'] = reformulated_solution
            
            combined_data.append(new_entry)
            
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
            continue
    
    # Sort by data_id to maintain consistent order
    combined_data.sort(key=lambda x: x.get('data_id', x.get('annot_id', '')))
    
    # Save the combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully combined {len(combined_data)} entries")
    print(f"Final output saved to {output_file}")

def create_reformulated_data(llm_engine_name="gpt-4o-mini", use_cache=False, max_workers=1, input_file="./combined_training_data.json", output_file="./reformulated_new_data.json", individual_files_dir="./raw"):
    # Import and create LLM engine
    from models.engines.factory import create_llm_engine
    llm_engine = create_llm_engine(llm_engine_name, use_cache)
    
    # Use the provided file paths
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return
    
    print(f"Processing {len(data)} entries...")
    
    # Process the data
    processed_data = []
    
    if max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_entry, llm_engine, entry, i, individual_files_dir)
                for i, entry in enumerate(data)
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing entries"):
                result = future.result()
                processed_data.append(result)
    else:
        for i, entry in enumerate(tqdm(data, desc="Processing entries")):
            result = process_entry(llm_engine, entry, i, individual_files_dir)
            processed_data.append(result)
    
    print(f"Successfully processed {len(processed_data)} entries")
    print(f"Individual files saved to {individual_files_dir}/")
    
    # After all files are generated, combine JSON files from raw directory
    combine_json_files_from_raw(individual_files_dir, output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4o-mini", help="LLM engine to use for enhancement")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Use cache for LLM calls")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--input_file", type=str, default="./raw/combined_training_data.json", help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default="./reformulated_new_data.json", help="Path to output JSON file")
    parser.add_argument("--individual_files_dir", type=str, default="./raw", help="Directory to save individual files")
    args = parser.parse_args()
    
    create_reformulated_data(args.llm_engine_name, args.use_cache, args.max_workers, args.input_file, args.output_file, args.individual_files_dir)
