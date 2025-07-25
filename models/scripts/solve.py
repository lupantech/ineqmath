import os
import sys
import json
import argparse
import random
import time
import atexit
import subprocess
from typing import List, Dict, Any
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import concurrent.futures
from tqdm import tqdm


class ProblemSolver:
    
    BOUND_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is $C=X$', where X is your calculated numerical bound value. Example: 'The answer is $C=1$'."
    RELATION_HINT_PROMPT = "Task description: Please solve the problem with clear, rigorous, and logically sound steps. At the end of your response, state your answer in exactly this format: 'The answer is (Letter) Symbol', where Letter is one of the given options. Example: 'The answer is (A) $\\leq$'."
    
    def __init__(self, llm_engine_name: str, use_cache: bool = False, vllm_config_path: str = None):
        from models.engines.factory import create_llm_engine
        self.vllm_config_path = vllm_config_path
        self.engine_name = llm_engine_name
        self.vllm_server_process = None
        # if vllm, set up the vllm server
        if llm_engine_name.startswith("vllm-"):
            self.setup_vllm_server()
            # Register cleanup function to be called when script exits
            atexit.register(self.cleanup)
        self.llm_engine = create_llm_engine(llm_engine_name, use_cache)
        
    def cleanup(self):
        """Clean up resources, especially VLLM server process"""
        if hasattr(self, 'vllm_server_process') and self.vllm_server_process is not None:
            print("Terminating VLLM server...")
            try:
                self.vllm_server_process.terminate()
                # Wait a bit for graceful termination
                try:
                    self.vllm_server_process.wait(timeout=10)
                    print("VLLM server terminated gracefully.")
                except subprocess.TimeoutExpired:
                    print("VLLM server didn't terminate gracefully, forcing kill...")
                    self.vllm_server_process.kill()
                    self.vllm_server_process.wait()
                    print("VLLM server killed.")
            except Exception as e:
                print(f"Error during VLLM server cleanup: {e}")
            finally:
                self.vllm_server_process = None

    def __del__(self):
        """Destructor to ensure cleanup happens"""
        self.cleanup()
        
    def build_query_o1(self, problem: str, prob_type: str) -> str:
        if prob_type == "bound":
            task_hint = f"{self.BOUND_HINT_PROMPT}"
        elif prob_type == "relation":
            task_hint = f"{self.RELATION_HINT_PROMPT}"
        else:
            raise ValueError(f"Unknown problem type: {prob_type}")
    
        query = f"Problem: {problem}\n\nHint: {task_hint}"
        return query

    def setup_vllm_server(self) -> None:
        # Check if vllm is installed
        try:
            import vllm
        except ImportError:
            raise ImportError("If you'd like to use VLLM models, please install the vllm package by running `pip install vllm`.")
        
        # Validate config path if provided
        if self.vllm_config_path is not None and not os.path.exists(self.vllm_config_path):
            raise ValueError(f"VLLM config path does not exist: {self.vllm_config_path}")
            
        # Start the VLLM server
        command = ["vllm", "serve", self.engine_name.replace("vllm-", ""), "--port", "8888", "--host", "0.0.0.0"]
        if self.vllm_config_path is not None:
            command = ["vllm", "serve", "--config", self.vllm_config_path, "--port", "8888", "--host", "0.0.0.0"]

        import threading
        import queue
        
        # Use queues to handle output from both stdout and stderr without deadlocks
        def enqueue_output(pipe, queue_obj):
            for line in iter(pipe.readline, ''):
                if line:
                    queue_obj.put(line)
            pipe.close()
        
        vllm_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Create queues and threads for non-blocking reading
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()
        
        stdout_thread = threading.Thread(target=enqueue_output, args=(vllm_process.stdout, stdout_queue))
        stderr_thread = threading.Thread(target=enqueue_output, args=(vllm_process.stderr, stderr_queue))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        print("Starting VLLM server...")
        startup_complete = False
        
        while not startup_complete:
            # Check if process has terminated
            if vllm_process.poll() is not None:
                print("VLLM server process terminated unexpectedly. Please check the output above for more information.")
                break
            
            # Read from stdout queue
            try:
                while True:
                    output = stdout_queue.get_nowait().strip()
                    if output:
                        print("VLLM server:", output)
                        if "Application startup complete." in output:
                            startup_complete = True
                            break
            except queue.Empty:
                pass
            
            # Read from stderr queue
            try:
                while True:
                    error = stderr_queue.get_nowait().strip()
                    if error:
                        print("VLLM server:", error)
                        if "Application startup complete." in error:
                            startup_complete = True
                            break
            except queue.Empty:
                pass
            
            if startup_complete:
                print("VLLM server started successfully.")
                break
                
            time.sleep(0.1)  # Short sleep to prevent busy waiting

        self.vllm_server_process = vllm_process
    
    def build_query(self, problem: str, prob_type: str, task_prompt: str = "", shot_num: int = 0) -> str:
        if prob_type == "bound":
            task_hint = f"{self.BOUND_HINT_PROMPT}\n\n"
        elif prob_type == "relation":
            task_hint = f"{self.RELATION_HINT_PROMPT}\n\n"
        else:
            raise ValueError(f"Unknown problem type: {prob_type}")

        task_prompt = f"{task_prompt}\n\n" if task_prompt else ""
        
        if shot_num == 0:
            demonstrations = ""
        else:
            # TODO: Implement this
            demonstrations = ""  

        query = f"{task_hint}{task_prompt}{demonstrations}Problem: {problem}\n\nSolution:"
        return query

    def build_md_text(self, problem: str, query: str, response: str) -> str:
        text = f"""
## Problem

{problem}

## Query

{query}

## Response

{response}
"""
        return text.strip()

    def process_problem(self, data: Dict[str, Any], output_dir: str, task_prompt: str, shot_num: int, **kwargs) -> None:
        data_id = data["annot_id"] if "annot_id" in data else data["data_id"] # NOTE

        try:
            problem = data["problem"]
            prob_type = data["type"]
            if "o1" in self.engine_name:
                query = self.build_query_o1(problem, prob_type)
            else:
                query = self.build_query(problem, prob_type, task_prompt, shot_num)
            response = self.llm_engine(query, **kwargs)
            if not response:
                print('response is empty')
            if type(response) == str:
                result = {**data, "prompt": query, "response": response, "success": True, "error": None}
            else:
                print(f"Response of problem {data_id} is not a string. Response: {response}")
                result = {**data, "prompt": query, "response": "", "success": False, "error": f"Response is not a string. Response: {response}"}
        except Exception as e:
            print(f"Error processing problem {data_id}: {str(e)}")
            result = {**data, "query": query, "response": "", "success": False, "error": str(e)}

        # print(f"Saving results for {data_id}...")
        self._save_results(result, output_dir, problem, query)

    def _save_results(self, result: Dict, output_dir: str, problem: str, query: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        data_id = result["annot_id"] if "annot_id" in result else result["data_id"] # NOTE
        
        # Save JSON result
        json_path = os.path.join(output_dir, f"{data_id}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        # Save markdown
        md_text = self.build_md_text(problem, query, result["response"])
        md_path = os.path.join(output_dir, f"{data_id}.md")
        with open(md_path, "w") as f:
            f.write(md_text)
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", type=str, default="../../data/final_data_250320/data/test_data_241127.json")
    parser.add_argument("--llm_engine_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--task_prompt", type=str,  default="")
    parser.add_argument("--prompt_dir", type=str, default="prompts")
    parser.add_argument("--shot_num", type=int, default=0)
    parser.add_argument("--test_num", type=int, default=1, help="-1 means all.")
    parser.add_argument("--run_label", type=str, default="exp1")
    parser.add_argument("--output_path", type=str, default="../../results/baselines_test_data_250320")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--dev", action="store_true", default=False, help="Use full dataset. If not set, start from index 100.")
    parser.add_argument("--vllm_config_path", type=str, default=None, help="Path to VLLM config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load test data
    if os.path.exists(args.test_data_path):
        with open(args.test_data_path, "r") as f:
            test_data: List[Dict[str, Any]] = json.load(f)
    else:
        print(f"File {args.test_data_path} not found, loading from HuggingFace dataset")
        # Login using e.g. `huggingface-cli login` to access this dataset
        ds = load_dataset("AI4Math/IneqMath")
        test_data = [dict(item) for item in ds['test']]  # Convert Dataset to list of dicts

    # Initialize solver
    solver = ProblemSolver(args.llm_engine_name, args.use_cache, args.vllm_config_path)
    
    try:    
        if args.test_num > 0:
            random.seed(42)  # Set seed for reproducibility
            test_data = random.sample(test_data, min(args.test_num, len(test_data)))
        print(f"We have {len(test_data)} test cases in total.")
        output_dir = os.path.join(args.output_path, args.run_label, "raw")
        # Skip if already processed successfully
        test_data_to_process = []
        for data in test_data:
            data_id = data["annot_id"] if "annot_id" in data else data["data_id"] # NOTE
            if os.path.exists(os.path.join(output_dir, f"{data_id}.json")):
                # print(output_dir)
                print(f"Skipping {data_id}: already processed successfully")
            else:
                test_data_to_process.append(data)
        print(f"Processing {len(test_data_to_process)} test cases...")

        # Create kwargs dictionary with additional arguments
        kwargs = {"max_tokens": args.max_tokens}
        
        if args.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(solver.process_problem, data, output_dir, args.task_prompt, args.shot_num, **kwargs) for data in test_data_to_process]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    future.result()
        else:
            for data in test_data_to_process:
                solver.process_problem(data, output_dir, args.task_prompt, args.shot_num, **kwargs)

        print("Done!")
    
    finally:
        # Ensure cleanup happens
        solver.cleanup()
