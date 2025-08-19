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
    Create transformation prompt to convert inequality problems into bound prediction problems
    """
    prompt = f"""**Task:** Transform the given inequality problem into a bound prediction problem by introducing a constant C and determining its optimal value.

**Instructions:**

1. Analyze the original problem, focusing on its structure and potential for transformation.
2. Introduce a constant $C$ by either replacing an existing constant or creating a new relationship between expressions.
3. Determine whether to find the minimal or maximal value of $C$ that satisfies the inequality for all relevant variables.
4. Consider factors such as homogeneity, existing constraints, and the domain of variables (e.g., positive reals, all reals).
5. Ensure the rephrased problem maintains the mathematical essence and constraints of the original.
6. **Ensure that the optimal value of C is a real number (not an expression involving variables).**

**Output format:**

Provide your response in the following structure:

<Analysis>: Concise explanation of key features and transformation approach.
<Conclusion>: YES or NO, followed by a brief summary of the transformation.
<Rephrased Problem>: Transformed problem statement, focusing on finding the optimal $C$.
<Answer>: $C = <numerical_value>$ where the value is a specific real number.

**Key considerations:**
1. For double inequalities, attempt to rephrase as a single bound prediction problem when possible.
2. In homogeneous inequalities, focus on the ratios between variables rather than their absolute values.
3. Incorporate any existing constraints into the rephrased version of the problem.
4. Clearly specify the domain of the variables in the rephrased problem statement.
5. Ensure that the rephrased problem is logically equivalent to the original.
6. If the side containing C is larger than the other side (e.g., f(x) ≤ Cg(x)), please let the question to be finding the smallest C that satisfies the inequality. If the side containing C is smaller than the other side (e.g., C ≤ f(x)), please let the question to finding the largest C that satisfies the inequality.
7. **The final answer must be a specific real number, not an algebraic expression or formula. (such as $n$ is not allowed)**

**Example:**

Original problem:
Let $a, b, c \\in \\mathbb{{R}}^{{+}}$. Prove the inequality
$$
\\frac{{a b c}}{{(1+a)(a+b)(b+c)(c+16)}} \\leq \\frac{{1}}{{81}}
$$

<Analysis>: To turn this into a bound prediction problem, we can focus on the following steps:
1. The left side is a rational expression that is always positive for $a, b, c \\in \\mathbb{{R}}^{{+}}$.
2. The right side is a fixed constant $\\frac{{1}}{{81}}$.
3. We replace the constant $\\frac{{1}}{{81}}$ with a variable $C$ and ask: What is the smallest $C$ such that the inequality holds for all positive $a, b, c$?
4. This approach allows us to determine the tightest possible upper bound for the left-hand expression.
5. If we find the smallest $C$ that works, we prove the original inequality and show it's the best possible.

<Conclusion>: YES, the inequality can be rephrased as a bound prediction problem. By replacing the constant $\\frac{{1}}{{81}}$ with a variable $C$, we can determine the tightest upper bound for the given rational expression, effectively proving the original inequality and demonstrating its optimality.

<Rephrased problem>:
Determine the minimal constant $C$ such that the following inequality holds for all $a, b, c \\in \\mathbb{{R}}^{{+}}$:
$$
\\frac{{abc}}{{(1+a)(a+b)(b+c)(c+16)}} \\leq C.
$$

<Answer>: $C = \\frac{{1}}{{81}}$.

**Now, please rewrite the following problem:**

Original problem: {problem}"""

    return prompt

def get_transformed_problem(llm_engine, problem, max_retries=3):
    """
    Use LLM engine to transform inequality problem into bound prediction problem
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
    Clean the answer by ensuring it has proper dollar sign formatting
    For example: 
    - $$C = 3$$ becomes $C = 3$
    - C = 1 becomes $C = 1$
    """
    if isinstance(answer, str):
        answer = answer.strip()
        # Remove double dollar signs from both sides and replace with single
        answer = re.sub(r'^\$\$(.+)\$\$$', r'$\1$', answer)
        
        # If answer doesn't have dollar signs on both sides, add them
        if not (answer.startswith('$') and answer.endswith('$')):
            answer = f'${answer}$'
    return answer

def process_single_problem(llm_engine, problem):
    """
    Process a single problem by transforming it into a bound prediction problem
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
        "type": "bound",
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
    Process a single entry and transform it into a bound prediction problem
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
            'type': 'bound',
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
            'type': 'bound',
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
                'type': data.get('type', 'bound'),
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
                           output_file="./raw/reformulated_bound_data.json", 
                           individual_files_dir="./raw/bound_problems"):
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
    
    parser = argparse.ArgumentParser(description="Transform inequality problems into bound prediction problems")
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4o-mini", help="LLM engine to use for transformation")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Use cache for LLM calls")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--input_file", type=str, default="./numina_ineq_2k_sampled.json", help="Input JSON file path")
    parser.add_argument("--output_file", type=str, default="./raw/reformulated_bound_data.json", help="Output JSON file path")
    parser.add_argument("--individual_files_dir", type=str, default="./raw/bound_problems", help="Directory to save individual markdown and JSON files")
    args = parser.parse_args()
    
    create_reformulated_data(
        args.llm_engine_name, 
        args.use_cache, 
        args.max_workers,
        args.input_file,
        args.output_file,
        args.individual_files_dir
    )
