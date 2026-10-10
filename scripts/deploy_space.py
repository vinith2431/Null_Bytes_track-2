"""Deploy the Aegis web UI to a Hugging Face Space (Docker SDK, free CPU).

    hf auth login                                   # once, paste a WRITE token from huggingface.co/settings/tokens
    python -m scripts.deploy_space <hf-user>/<space-name>

Uploads only files tracked by git plus the trained Q-Gate models and the M3 training CSV (the models are
gitignored). Never uploads .env. Then set the API key in the Space: Settings > Variables and secrets."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["models/qgate_quantum_domain.joblib", "models/qgate_quantum_hf.joblib", "models/qgate_rbf_hf.joblib", "models/qgate_rbf_hf_8d_embedder.joblib",
         "data/hf_prompt_injection/train.csv"]          # walkthrough's "looks like" examples come from here
HEADER = """---
title: Aegis
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

"""


def main(repo_id: str):
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = [f for f in tracked if f and (ROOT / f).is_file()] + EXTRA
    missing = [f for f in EXTRA if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"missing {missing}: run python -m scripts.export_hf_dataset and python -m scripts.train_qgate_semantic")
    assert ".env" not in files
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            (Path(tmp) / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, Path(tmp) / f)
        readme = Path(tmp) / "README.md"            # Spaces read their settings from the README front matter
        readme.write_text(HEADER + readme.read_text(encoding="utf-8"), encoding="utf-8")
        api = HfApi()
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=tmp,
                          commit_message="Deploy Aegis web UI", delete_patterns=["*"])
    print(f"uploaded {len(files)} files. Building at https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python -m scripts.deploy_space <hf-user>/<space-name>")
    main(sys.argv[1])
