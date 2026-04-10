import argparse
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

models_dir=Path(__file__).parent.parent / "models"
models = {
    "phi": {"repo_id": "bartowski/microsoft_Phi-4-mini-instruct-GGUF", "revision": "7ff82c2aaa4dde30121698a973765f39be5288c0", "type": "gguf"},
    "llama": {"repo_id": 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', "revision": "bf5b95e96dac0462e2a09145ec66cae9a3f12067", "type": "gguf"},
    "qwen": {"repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "revision": '8911e8a47f92bac19d6f5c64a2e2095bd2f7d031', "type": "gguf"},
    "gliner": {"repo_id": "urchade/gliner_large-v2.1", "revision": "abd49a1f1ebc12af1be84d06f6848221cf96dcad", "type": "snapshot"},
}

def download_one(name, cfg, force=False):
    if cfg["type"] == "snapshot":
        local_dir = models_dir / name
        if (local_dir / "config.json").exists() and not force:
            print(f"{name} already downloaded")
            return True
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"{name} downloading")
        try:
            snapshot_download(repo_id=cfg["repo_id"], revision=cfg["revision"], repo_type="model", local_dir=local_dir)
        except Exception as e:
            print(f"{name} fetch failed {e}")
            return False
    else:
        fname=cfg["repo_id"].split("/")[1].replace("-GGUF", "") + "-Q4_K_M.gguf"
        out_file = models_dir / name / fname
        if out_file.exists() and not force:
            print(f"{name} already downloaded")
            return True
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name} downloading")
        try:
            hf_hub_download(repo_id=cfg["repo_id"],filename=fname, repo_type="model",revision=cfg["revision"], local_dir=out_file.parent)
        except Exception as e:
            print(f"{name} fetch failed {e}")
            return False
        if not out_file.exists():
            print(f"{name} file not found after download")
            return False
    print(f"{name} done")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=[*models, "all"], default=["all"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    targets= list(models) if "all" in args.models else args.models
    ok=all(download_one(name, models[name], force=args.force) for name in targets)
    if not ok:
        print("one or more models failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
