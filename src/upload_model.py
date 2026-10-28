from huggingface_hub import HfApi

MODEL_DIR = "models/final_model"
REPO_NAME = "complaint-triage-distilbert"


def upload() -> None:
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{REPO_NAME}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=MODEL_DIR, repo_id=repo_id, repo_type="model")
    print("Uploaded to:", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload()