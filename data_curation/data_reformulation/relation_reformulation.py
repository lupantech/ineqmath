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
class TransformedProblem(BaseModel):
    analysis: str
    conclusion: str
    rephrased_problem: str
    answer: str

def get_transformation_prompt(problem):
    """
    Create transformation prompt to convert inequality problems into relation prediction problems
    """
    prompt = f"""**Task:** Transform the given inequality proof problem into a relation prediction problem.

**Instructions:**

1. Analyze the original problem, identifying key components such as variables, domains, conditions, and the main inequality.
2. Rephrase the problem by maintaining the original expressions and replacing the relation symbol with a blank to be filled.
3. Preserve any additional conditions or constraints from the original problem in your rephrased version.
4. Change the task from "Prove" to "Determine the correct inequality relation to fill in the blank."
5. Provide a set of options for the relation, always including (A) $\\leq$, (B) $\\geq$, (C) $=$, (D) $<$, (E) $>$, and (F) "None of the above".
6. Determine the correct answer based on your modification and analysis.

**Output format:**

Provide your response in the following structure:

<Analysis>: Detailed step-by-step analysis of the original problem and your approach to rephrasing it.

<Conclusion>: YES or NO, followed by a brief explanation of whether and how the problem can be effectively rephrased.

<Rephrased Problem>: 

Transformed problem statement.

Options:
(A) $\\leq$ 
(B) $\\geq$
(C) $=$ 
(D) $<$ 
(E) $>$ 
(F) None of the above

<Answer>: Option selected (Option letter and the relation symbol).

**Key considerations:**
1. Maintain the original mathematical expressions and any given conditions as much as possible.
2. Ensure the rephrased problem captures the essence and complexity of the original problem.
3. For problems with multiple inequalities, focus on one main inequality for the relation prediction task.
4. When dealing with complex fractions or expressions, keep them intact to maintain the problem's difficulty level.
5. If the relation depends on specific values of the variables or cannot be definitively determined, consider using "None of the above" as the correct answer.

**Example:**

Original problem:
Let $a, b, c \\in \\mathbb{{R}}^{{+}}$. Prove the inequality
$$
\\frac{{a b c}}{{(1+a)(a+b)(b+c)(c+16)}} \\leq \\frac{{1}}{{81}}
$$

<Analysis>: To rephrase it to a relation prediction problem, we can focus on the following steps:
1. The original problem is a proof task for an inequality involving positive real numbers $a$, $b$, and $c$.
2. The left side of the inequality is a complex fraction $\\frac{{a b c}}{{(1+a)(a+b)(b+c)(c+16)}}$.
3. The right side is a constant fraction $\\frac{{1}}{{81}}$.
4. The original inequality uses the "less than or equal to" ($\\leq$) relation, which needs to hold for all positive real values of $a$, $b$, and $c$.
5. We can transform the proof task into determining the correct relation between the left and right sides of the inequality.

<Conclusion>: YES, the inequality can be effectively rephrased as a relation prediction problem.

<Rephrased Problem>:
Let $a, b, c \\in \\mathbb{{R}}^{{+}}$. Consider the following inequality:
$$
\\frac{{a b c}}{{(1+a)(a+b)(b+c)(c+16)}} \\quad (\\quad) \\quad \\frac{{1}}{{81}} .
$$
Determine the correct inequality relation to fill in the blank.

Options:
(A) $\\leq$ 
(B) $\\geq$
(C) $=$ 
(D) $<$ 
(E) $>$ 
(F) None of the above

<Answer>: (A) $\\leq$ 

**Now, please rewrite the following problem:**

Original problem: {problem}"""

    return prompt

def get_transformed_problem(llm_engine, problem, max_retries=3):
    """
    Use LLM engine to transform inequality problem into relation prediction problem
    """
    prompt = get_transformation_prompt(problem)
    
    for attempt in range(max_retries):
        try:
            response = llm_engine(prompt, max_tokens=10000, response_format=TransformedProblem)
            
            if (response.analysis and response.conclusion and 
                response.rephrased_problem and response.answer):
                return response
            else:
                print(f"Attempt {attempt + 1}: Got incomplete response")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to transform problem after {max_retries} attempts.")
                return None
    
    return None

def clean_answer(answer):
    """
    Clean the answer by removing any extra formatting
    For relation prediction problems, normalize to include the option letter AND the relation symbol,
    e.g., "(A) $\\leq$", not just the letter.
    """
    if not isinstance(answer, str):
        return answer

    raw = answer.strip()
    lower = raw.lower()

    # Standardized outputs
    choice_mapping = {
        'A': "(A) $\\leq$",
        'B': "(B) $\\geq$",
        'C': "(C) $=$",
        'D': "(D) $<$",
        'E': "(E) $>$",
        'F': "(F) None of the above",
    }

    # If it already matches one of the standardized forms, return as-is (after minimal trim)
    for value in choice_mapping.values():
        if value.replace(' ', '') in raw.replace(' ', ''):
            return value

    # Detect "None of the above" → F
    if "none of the above" in lower:
        return choice_mapping['F']

    # Detect relation symbol (prefer longer tokens first)
    symbol_to_letter = [
        (['≤', 'leq', '<='], 'A'),
        (['≥', 'geq', '>='], 'B'),
        (['='], 'C'),
        (['<'], 'D'),  # Keep after '<=' check above
        (['>'], 'E'),  # Keep after '>=' check above
    ]

    detected_letter_from_symbol = None
    for tokens, letter in symbol_to_letter:
        for tok in tokens:
            if tok in lower:
                detected_letter_from_symbol = letter
                break
        if detected_letter_from_symbol:
            break

    # Detect explicit option letter patterns like "(A)" if symbol not found
    detected_letter_from_text = None
    m = re.search(r"\(([A-Fa-f])\)", raw)
    if m:
        detected_letter_from_text = m.group(1).upper()
    else:
        # Try looser patterns: 'Option A', 'Answer: A', or standalone A/B/C/D/E/F bounded by non-letters
        m2 = re.search(r"\b([A-Fa-f])\b", raw)
        if m2:
            detected_letter_from_text = m2.group(1).upper()

    # Prefer symbol-derived letter; fall back to text-derived letter
    final_letter = detected_letter_from_symbol or detected_letter_from_text
    if final_letter and final_letter in choice_mapping:
        return choice_mapping[final_letter]

    # If nothing matched, return the raw answer unchanged
    return raw

def process_single_problem(llm_engine, problem):
    """
    Process a single problem by transforming it into a relation prediction problem
    """
    transformed = get_transformed_problem(llm_engine, problem)
    
    if transformed is None:
        return None
    
    # Extract answer value from the response
    processed_answer = clean_answer(transformed.answer)
    
    return {
        'analysis': transformed.analysis,
        'conclusion': transformed.conclusion,
        'rephrased_problem': transformed.rephrased_problem,
        'answer': processed_answer,
        'solution': f"{transformed.analysis}\n\n{transformed.conclusion}\n\n{transformed.rephrased_problem}"
    }

def create_markdown_content(original_problem, transformed_data):
    """
    Create markdown content for the processed entry
    """
    markdown_content = f"""# Problem Entry

## Original Problem
{original_problem}

## Analysis
{transformed_data['analysis']}

## Conclusion
{transformed_data['conclusion']}

## Rephrased Problem
{transformed_data['rephrased_problem']}

## Answer
{transformed_data['answer']}
"""
    return markdown_content

def save_individual_files(entry_data, output_dir, original_problem, transformed_data, entry_id):
    """
    Save individual markdown and JSON files for each entry
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save markdown file
    markdown_content = create_markdown_content(original_problem, transformed_data)
    md_path = os.path.join(output_dir, f"{entry_id}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Save JSON file
    json_data = {
        "data_id": entry_id,
        "type": "relation",
        "problem": transformed_data['rephrased_problem'],
        "solution": transformed_data['solution'],
        "answer": transformed_data['answer'],
        "original_problem": original_problem,
        "analysis": transformed_data['analysis'],
        "conclusion": transformed_data['conclusion'],
        "original_entry": entry_data
    }
    json_path = os.path.join(output_dir, f"{entry_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def process_entry(llm_engine, entry, entry_index, individual_files_dir):
    """
    Process a single entry and transform it into a relation prediction problem
    """
    # Extract required fields - assuming the structure has a 'problem' field
    problem = entry.get('problem', entry.get('text', ''))
    entry_id = entry.get('id', entry.get('data_id', f"entry_{entry_index}"))
    
    # Skip if problem is missing
    if not problem:
        print(f"Skipping entry {entry_index + 1}: missing problem field")
        return None
    
    # Check if this entry has already been processed
    json_path = os.path.join(individual_files_dir, f"{entry_id}.json")
    md_path = os.path.join(individual_files_dir, f"{entry_id}.md")
    
    if os.path.exists(json_path) and os.path.exists(md_path):
        print(f"Entry {entry_index + 1} (ID: {entry_id}) already processed, skipping...")
        # Load and return the existing processed entry
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        return {
            'data_id': entry_id,
            'type': 'relation',
            'problem': existing_data['problem'],
            'solution': existing_data['solution'],
            'answer': existing_data['answer']
        }
    
    try:
        # Transform the problem
        transformed_data = process_single_problem(llm_engine, problem)
        
        if transformed_data is None:
            print(f"Failed to transform entry {entry_index + 1} (ID: {entry_id})")
            return None
        
        print(f"Successfully transformed entry {entry_index + 1} (ID: {entry_id})")
        
        # Save individual markdown and JSON files
        save_individual_files(
            entry, 
            individual_files_dir, 
            problem,
            transformed_data,
            entry_id
        )
        
        # Return the processed entry in the expected format
        return {
            'data_id': entry_id,
            'type': 'relation',
            'problem': transformed_data['rephrased_problem'],
            'solution': transformed_data['solution'],
            'answer': transformed_data['answer']
        }
        
    except Exception as e:
        print(f"Error processing entry {entry_index + 1} (ID: {entry_id}): {str(e)}")
        return None

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
                
            # Create new entry with the required fields
            new_entry = {
                'data_id': data.get('data_id', ''),
                'type': data.get('type', 'relation'),
                'problem': data.get('problem', ''),
                'solution': data.get('original_entry', {}).get('solution', ''),
                'answer': data.get('answer', '')
            }
            
            combined_data.append(new_entry)
            
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}")
            continue
    
    # Sort by data_id to maintain consistent order
    combined_data.sort(key=lambda x: x.get('data_id', ''))
    
    # Save the combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully combined {len(combined_data)} entries")
    print(f"Final output saved to {output_file}")

def create_reformulated_data(llm_engine_name="gpt-4o-mini", use_cache=False, max_workers=1, 
                            input_file="./numina_ineq_2k_sampled.json", 
                            output_file="./raw/reformulated_relation_data.json", 
                            individual_files_dir="./raw/relation_problems"):
    # Import and create LLM engine
    from models.engines.factory import create_llm_engine
    llm_engine = create_llm_engine(llm_engine_name, use_cache)
    
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
                if result is not None:
                    processed_data.append(result)
    else:
        for i, entry in enumerate(tqdm(data, desc="Processing entries")):
            result = process_entry(llm_engine, entry, i, individual_files_dir)
            if result is not None:
                processed_data.append(result)
    
    print(f"Successfully processed {len(processed_data)} entries")
    print(f"Individual files saved to {individual_files_dir}/")
    
    # After all files are generated, combine JSON files from raw directory
    combine_json_files_from_raw(individual_files_dir, output_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4o-mini", help="LLM engine to use for transformation")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Use cache for LLM calls")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--input_file", type=str, default="./numina_ineq_2k_sampled.json", help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default="./raw/reformulated_relation_data.json", help="Path to output JSON file")
    parser.add_argument("--individual_files_dir", type=str, default="./raw/relation_problems", help="Directory to save individual processed files")
    args = parser.parse_args()
    
    create_reformulated_data(args.llm_engine_name, args.use_cache, args.max_workers, 
                           args.input_file, args.output_file, args.individual_files_dir)
