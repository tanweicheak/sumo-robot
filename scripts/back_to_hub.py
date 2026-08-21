"""
scripts.backup_to_hub

Phase: pre/post-training safety net (item 5, Layer 2)
Purpose: A RunPod Network Volume (Layer 1 - a config/infra choice, not code) covers
    the core risk of losing everything on pod termination. This is Layer 2:
    redundancy specifically for the FINAL, most expensive-to-regenerate artifacts
    (LoRA adapters, merged fp16, quantized GGUF), pushed somewhere fully outside
    RunPod's ecosystem - so even a Network Volume misconfiguration or account issue
    doesn't leave you with nothing. Not a substitute for Layer 1; both together.

    Deliberately narrow scope: this pushes model artifacts, not the JSONL logs
    (training_pairs.jsonl, prompt_history.jsonl, mcts_calibration.jsonl) - those
    are better served by Layer 1's Network Volume (many small files, not really
    what HF Hub repos are built for). If a specific run's raw logs are also
    precious enough to want off-RunPod, run this with --extra-dir pointing at
    that run's checkpoint directory too.

Usage:
    huggingface-cli login   # once, interactively
    python -m scripts.backup_to_hub \\
        --repo-id your-username/sumo-sbso-benchmark2 \\
        --adapter-dir checkpoints/benchmark2_full_sbso/lora_run/lora_adapters \\
        --merged-dir checkpoints/benchmark2_full_sbso/merged_fp16 \\
        --gguf-dir checkpoints/benchmark2_full_sbso/gguf

    # Push whichever of the three actually exist yet - none are required, and a
    # missing directory is skipped with a note, not a crash. Run this once after
    # each real artifact-producing step, not only at the very end - an adapter
    # worth backing up exists long before the GGUF export does.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def backup_directory(repo_id: str, local_dir: str | Path, path_in_repo: str, private: bool = True) -> bool:
    """Uploads local_dir to path_in_repo within repo_id. Returns False (does not
    raise) if local_dir doesn't exist yet, so a partial pipeline (e.g. adapters
    exist, GGUF doesn't yet) can still back up what it has."""
    local_dir = Path(local_dir)
    if not local_dir.exists():
        print(f"[backup_to_hub] SKIP {path_in_repo}: {local_dir} does not exist yet")
        return False

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    print(f"[backup_to_hub] uploading {local_dir} -> {repo_id}/{path_in_repo} ...")
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, path_in_repo=path_in_repo)
    print(f"[backup_to_hub] done: {path_in_repo}")
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Back up training artifacts to a private HF Hub repo.")
    p.add_argument("--repo-id", required=True, help="e.g. your-username/sumo-sbso-benchmark2")
    p.add_argument("--adapter-dir", default=None, help="LoRA adapters directory")
    p.add_argument("--merged-dir", default=None, help="Merged fp16 model directory")
    p.add_argument("--gguf-dir", default=None, help="Quantized/exported GGUF directory")
    p.add_argument("--extra-dir", action="append", default=[],
                    help="Additional directory to back up, format LOCAL_PATH:PATH_IN_REPO. "
                         "Repeatable - pass multiple times for multiple extra directories.")
    p.add_argument("--public", action="store_true", help="Make the repo public instead of private (default: private)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    private = not args.public

    targets = [
        (args.adapter_dir, "lora_adapters"),
        (args.merged_dir, "merged_fp16"),
        (args.gguf_dir, "gguf"),
    ]
    for extra in args.extra_dir:
        local_path, _, path_in_repo = extra.partition(":")
        if not path_in_repo:
            raise SystemExit(f"--extra-dir must be LOCAL_PATH:PATH_IN_REPO, got: {extra!r}")
        targets.append((local_path, path_in_repo))

    any_uploaded = False
    for local_dir, path_in_repo in targets:
        if local_dir is None:
            continue
        any_uploaded = backup_directory(args.repo_id, local_dir, path_in_repo, private=private) or any_uploaded

    if not any_uploaded:
        print("[backup_to_hub] nothing was uploaded - none of the given directories exist yet. "
              "Run this again once an artifact-producing step has completed.")


if __name__ == "__main__":
    main()