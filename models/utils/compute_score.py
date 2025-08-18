import os
import sys
import json
import argparse
import re
from typing import List, Dict, Any

import random
random.seed(42)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import concurrent.futures
from tqdm import tqdm

from pydantic import BaseModel

# Executor: EquivalenceCheck
class EquivalenceCheck(BaseModel):
    analysis: str
    equivalent: bool

CHOICES_DICT = {
    "A": "(A) $\leq$", 
    "B": "(B) $\geq$", 
    "C": "(C) $=$", 
    "D": "(D) $<$",
    "E": "(E) $>$",
    "F": "(F) None of the above",
}
CHOICES = list(CHOICES_DICT.values())
    
def locate_answer(prediction: str) -> str:
    # Locate the sentence in the prediction that contains the answer, which is formatted as "The answer is X"
    # try:
    #     # locate the last index of "answer is":
    #     prediction = prediction.replace("final answer is", "answer is")
    #     answer_is_idx = prediction.lower().rfind("answer is")
    #     # print(f"answer_is_idx: {answer_is_idx}")
    #     if answer_is_idx == -1:
    #         return ""
        
    #     extraction = prediction[answer_is_idx:].strip()
    #     extraction = extraction.replace("\(", "$")
    #     extraction = extraction.replace("\)", "$")
    #     extraction = re.sub(r'\s+', ' ', extraction).strip() # replace multiple whitespaces with a single whitespace
    #     return extraction if extraction else ""
    # except:
    #     return ""

    # Case 1: If prediction is None or empty, return empty string
    try:
        prediction = prediction.strip() if prediction else ""
    except:
        return ""
        # print("*"*50)
        # print(f"prediction: {prediction}")
        # print(f"type(prediction): {type(prediction)}")
        # raise ValueError(f"prediction is not a string: {prediction}")
    if prediction is None or prediction == "":
        return ""

    # Case 2: Return the first sentence that contains "answer is"
    # Define trigger phrases from most specific to least specific
    trigger_phrases = ["final answer is", "answer is", "final answer"]
    
    # Return on first match
    prediction_lower = prediction.lower()
    for phrase in trigger_phrases:
        idx = prediction_lower.rfind(phrase)
        if idx != -1:
            extraction = prediction[idx:].strip()
            extraction = extraction.replace("\(", "$")
            extraction = extraction.replace("\)", "$")
            extraction = re.sub(r'\s+', ' ', extraction).strip()
            return extraction
        
    # Case 3: Return the last three sentences
    sentences = prediction.split("\n")
    matched = sentences[-3:]
    return "\n".join(matched)


def extract_relation_answer(answer_sentence: str, args: argparse.Namespace):
    answer_sentence = re.sub(r'\s+', ' ', answer_sentence).strip()

    # Find the matching choice
    for choice in CHOICES:
        if choice in answer_sentence:
            return choice

    # Quick extraction for common choices
    if any(match in answer_sentence for match in ["(A)", "leq", "≤"]):
        return CHOICES_DICT["A"]
    elif any(match in answer_sentence for match in ["(B)", "geq", "≥"]):
        return CHOICES_DICT["B"]
    elif any(match in answer_sentence for match in ["(C)", "="]):
        return CHOICES_DICT["C"]
    elif any(match in answer_sentence for match in ["(D)", "<"]):
        return CHOICES_DICT["D"]
    elif any(match in answer_sentence for match in ["(E)", ">"]):
        return CHOICES_DICT["E"]
    elif any(match in answer_sentence for match in ["(F)", "none", "None"]):
        return CHOICES_DICT["F"]

    # If no match is found, use LLM to extract the answer
    examples = open(args.relation_prompt_path, "r").read()
    # print(examples)

    prompt = f"""
    You are an expert at extracting option letters (A, B, C, D, E, F) from answer sentences. 

    The options are given below:
    {CHOICES}

    Below are examples of sentences and the corresponding option letters:

    {examples}

    Now, extract the option letter from the following sentence:
    {answer_sentence}

    Make sure to return the option letter only, without any other characters.

    Answer:
    """
    choice_letter = local_llm_engine(prompt).strip()
    if choice_letter in CHOICES_DICT:
        return CHOICES_DICT[choice_letter]
    else:
        random_choice = random.choice(CHOICES)
        print(f"No matching choice found for {answer_sentence}")
        print(f"Returning random choice: {random_choice}")
        return random_choice

def verify_relation_answer(ground_truth: str, prediction: str, args: argparse.Namespace) -> bool:
    assert prediction in CHOICES, f"Prediction {prediction} is not in CHOICES"
    if ground_truth == prediction:
        return True
    else:
        return False

def extract_bound_answer(prediction: str, args: argparse.Namespace):
    # Case 1: Extract the answer between $$ symbols
    # replace "$\boxed{XYZ}$" with "$XYZ$"

    if args.bound_answer_quick_extraction:
        print("Using quick extraction to find the answer between $$ symbols.")
        pattern = r'\$(.*?)\$'  # Simplified pattern to match anything between $$ symbols
        match = re.search(pattern, prediction)
        if match:
            match_str = match.group(1).strip()
            # Remove \boxed{...} if the string starts and ends with it
            if match_str.startswith("\\boxed{") and match_str.endswith("}"):
                match_str = match_str[7:-1].strip()  # 7 is the length of "\boxed{"
            return match_str

            # # FIXME Remove \boxed{X} wrapper if present
            # boxed_pattern = r'\\boxed{(.*?)}' # greedy match
            # boxed_match = re.search(boxed_pattern, match_str, re.DOTALL)
            # if boxed_match:
            #     match_str = boxed_match.group(1).strip()
            # if match_str:
            #     return match_str

    # Use LLM to extract the answer
    examples = open(args.bound_prompt_path, "r").read()
    prompt = f"""
    You are an expert at extracting numbers from answer sentences. 

    Below are examples of sentences and the corresponding numbers:

    {examples}

    Now, extract the number from the following sentence:
    {prediction}

    Make sure to return the answer in a format as "C=<extracted_answer>", where <extracted_answer> is the extracted number or expression.

    Answer:
    """ 
    extracted_answer = local_llm_engine(prompt).strip()
    if extracted_answer:
        return extracted_answer

    # Others: Return empty string if no answer is found
    return ""


def verify_bound_answer_with_llm(ground_truth: str, prediction: str, args: argparse.Namespace) -> bool:
    examples = open(args.bound_verification_prompt_path, "r").read()
    
    prompt = f"""
    You are an expert at verifying mathematical expression equivalence. Analyze if two expressions are exactly equivalent by following these strict rules:

    Required Analysis Steps:
    1. Check if both expressions are valid mathematical forms
    2. If either expression is not mathematical (e.g., text or undefined), return False
    3. For numerical expressions:
       - Direct equality (e.g., 2 = 2) → True
       - Different representations of same value (e.g., 1/2 = 0.5, √4 = 2) → True
       - Decimal approximations vs exact values (e.g., 2π ≠ 6.28318) → False
    4. For algebraic expressions:
       - Must have clear, valid transformation path between forms
       - If transformation requires multiple non-obvious steps → False
       - Verify equivalence through algebraic proof when possible
       - For complex expressions, use techniques like squaring or substitution to verify

    Equivalence Criteria:
    - Must have exactly same deterministic value
    - Must be provably equivalent through valid mathematical operations
    - Different notations of same exact value are equivalent
    - Decimal approximations are NOT equivalent to exact expressions
    - No rounding or approximations allowed
    - If equivalence cannot be conclusively proven → False

    Example pairs and their analysis:
    {examples}

    Now analyze these expressions:
    Ground truth: {ground_truth}
    Prediction: {prediction}

    Provide step-by-step analysis first, then conclude with:
    Equivalent: [true/false]
    """
    
    try:
        result = local_llm_engine(prompt, response_format=EquivalenceCheck)
        print(f"Analysis: {result.analysis}")
        print(f"Equivalent: {result.equivalent}")
        # Look for "equivalent: true" or just "true" at the end of response
        return result.equivalent
    except Exception as e:
        print(f"Error in verify_bound_answer_with_llm: {e}")
        return False

def verify_bound_answer(ground_truth: str, prediction: str, args: argparse.Namespace) -> bool:
    """
    Verify if the extracted answer is equivalent to the ground truth.
    - First, clean the answer by removing $ and spaces.
    - If they are exactly the same, return True.
    - Otherwise, use LLM to verify if they are equivalent.
    """
    def clean_answer(answer: str):
        return answer.replace("$", "").replace(" ", "").strip()
    
    ground_truth = clean_answer(ground_truth)
    prediction = clean_answer(prediction)

    # Case 1: If they are exactly the same, return True (Strict case)
    if ground_truth == prediction:
        return True

    # Case 2: Use LLM to verify the answer further (Less strict)
    return verify_bound_answer_with_llm(ground_truth, prediction, args)
    
class ScoreComputer:
    def __init__(self, llm_engine_name: str, use_cache: bool = False):
        from models.engines.factory import create_llm_engine
        self.llm_engine = create_llm_engine(llm_engine_name, use_cache)
        global local_llm_engine
        local_llm_engine = self.llm_engine

    def evaluate_single_result(self, annot_id: str, result: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        problem = result.get("problem", "")
        prob_type = result["type"]
        ground_truth = result.get("answer", "")
        prediction = result.get("response", "")
        error = result.get("error", None)

        answer_sentence = locate_answer(prediction)

        if answer_sentence == "":
            print(f"No answer sentence found for {annot_id}")
            evaluation = {
                "ground_truth": ground_truth,
                "answer_sentence": "",
                "extracted_answer": "",
                "is_solved": False,
                "is_correct": False,
            }
        elif error:
            print(f"Error found for {annot_id}: {prediction}")
            evaluation = {
                "ground_truth": ground_truth,
                "answer_sentence": "",
                "extracted_answer": "",
                "is_solved": False,
                "is_correct": False,
            }
        else:
            if prob_type == "bound":
                extracted_answer = extract_bound_answer(answer_sentence, args)
                is_correct = verify_bound_answer(ground_truth, extracted_answer, args)
            elif prob_type == "relation":
                extracted_answer = extract_relation_answer(answer_sentence, args)
                is_correct = verify_relation_answer(ground_truth, extracted_answer, args)
            else:
                raise ValueError(f"Unknown problem type: {prob_type}")

            evaluation = {
                "ground_truth": ground_truth,
                "answer_sentence": answer_sentence,
                "extracted_answer": extracted_answer,
                "is_solved": True,
                "is_correct": is_correct,
            }

        return {
            annot_id: {
                **result,
                "evaluation": evaluation,
            }
        }

    def process_result(self, result_file: str, raw_result_dir: str, args: argparse.Namespace) -> Dict[str, Any]:
        # print(f"\nProcessing {annot_id}")
        annot_id = result_file.split(".")[0]
        
        with open(os.path.join(raw_result_dir, result_file), "r") as f:
            result = json.load(f)

        return self.evaluate_single_result(annot_id, result, args)

    @staticmethod
    def calculate_score(results: Dict[str, Any]) -> Dict[str, Any]:
        scores_init = {
                "all": {"total": 0, "correct": 0, "wrong": 0, "accuracy": 0.0, "empty_responses": 0},
                "bound": {"total": 0, "correct": 0, "wrong": 0, "accuracy": 0.0},
                "relation": {"total": 0, "correct": 0, "wrong": 0, "accuracy": 0.0},
            }
        scores = {}

        for test_id, result in results.items():
            prob_type = result["type"]         # "bound" or "relation"
            evaluation = result["evaluation"]
            data_split = result.get("data_split", "test")  # Default to test if not specified
            
            if data_split not in scores:
                scores[data_split] = scores_init.copy().copy()
            
            # Check if the response is empty
            if evaluation["answer_sentence"] == "":
                scores[data_split]["all"]["empty_responses"] += 1
            
            # Update counts for the specific data split
            scores[data_split]["all"]["total"] += 1
            scores[data_split][prob_type]["total"] += 1
            
            if evaluation["is_correct"]:
                scores[data_split]["all"]["correct"] += 1
                scores[data_split][prob_type]["correct"] += 1
            else:
                scores[data_split]["all"]["wrong"] += 1
                scores[data_split][prob_type]["wrong"] += 1

        # Calculate accuracy for all splits
        for split in scores.keys():
            for category in ["all", "bound", "relation"]:
                if scores[split][category]["total"] > 0:
                    scores[split][category]["accuracy"] = round(100 * scores[split][category]["correct"] / scores[split][category]["total"], 2)

        return scores

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--result_dir", type=str, default="")
    parser.add_argument("--run_label", type=str, default="exp1")
    parser.add_argument("--direct_input", action="store_true", default=False, help="If set, use --results_file directly and write scores in the same directory.")
    parser.add_argument("--results_file", type=str, default=None, help="Path to a results.json file to score when using --direct_input.")
    parser.add_argument("--relation_prompt_path", type=str, default="../prompts/answer_extraction_relation.md")
    parser.add_argument("--bound_prompt_path", type=str, default="../prompts/answer_extraction_bound.md")
    parser.add_argument("--bound_verification_prompt_path", type=str, default="../prompts/answer_verification_bound.md")
    parser.add_argument("--bound_answer_quick_extraction", action="store_true", default=False, help="If True, use quick extraction to find the answer between $$ symbols.")
    parser.add_argument("--max_workers", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_arguments()

    score_computer = ScoreComputer(args.llm_engine_name, args.use_cache)
    
    if args.direct_input:
        if not args.results_file:
            raise ValueError("--results_file must be provided when --direct_input is set.")
        input_results_file = args.results_file
        output_dir = os.path.dirname(args.results_file)
        output_scores_file = os.path.join(output_dir, "scores.json")
    else:
        output_dir = os.path.join(args.result_dir, args.run_label)
        input_results_file = os.path.join(output_dir, "results.json")
        output_scores_file = os.path.join(output_dir, "scores.json")

    if args.bound_answer_quick_extraction:
        output_scores_file = output_scores_file.replace(".json", "_quick_extraction.json")

    if not os.path.exists(input_results_file):
        raise FileNotFoundError(
            f"results.json not found at {input_results_file}. Please ensure your generation step has produced this file."
        )

    with open(input_results_file, "r") as f:
        loaded_results = json.load(f)

    # Build evaluated results in-memory (do not write results.json)
    evaluated_results: Dict[str, Any] = {}

    # Support either dict keyed by annot_id/data_id or a list of records
    if isinstance(loaded_results, dict):
        items_iterable = loaded_results.items()
    elif isinstance(loaded_results, list):
        items_iterable = [
            (str(item.get("annot_id", item.get("data_id"))), item) for item in loaded_results
        ]
    else:
        raise ValueError("results.json should be either a dict keyed by IDs or a list of result objects.")

    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(score_computer.evaluate_single_result, annot_id, result, args)
                for annot_id, result in items_iterable
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                evaluated_results.update(future.result())
    else:
        for annot_id, result in items_iterable:
            evaluated_results.update(score_computer.evaluate_single_result(annot_id, result, args))

    # Sort by numeric id if possible
    try:
        evaluated_results = dict(sorted(evaluated_results.items(), key=lambda x: int(x[0])))
    except Exception:
        evaluated_results = dict(sorted(evaluated_results.items(), key=lambda x: str(x[0])))

    scores = score_computer.calculate_score(evaluated_results)
    print(f"Score: {scores}")

    with open(output_scores_file, "w") as f:
        json.dump(scores, f, indent=4)

if __name__ == "__main__":
    main()