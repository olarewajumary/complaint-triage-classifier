from huggingface_hub import HfApi

REPO_ID = "maryolarewaju/complaint-triage-distilbert"


def list_files() -> None:
    info = HfApi().model_info(REPO_ID, files_metadata=True)
    for f in info.siblings:
        print(f.rfilename, f.size)


if __name__ == "__main__":
    list_files()