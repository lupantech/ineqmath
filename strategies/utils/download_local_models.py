import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download a model from HuggingFace Hub")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repository ID")
    parser.add_argument("--local_dir", required=True, help="Local directory to save the model")
    
    args = parser.parse_args()
    
    # Download the model to local_model directory
    model_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
    )
    
    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    main()