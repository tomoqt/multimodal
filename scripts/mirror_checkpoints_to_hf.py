#!/usr/bin/env python3
"""Mirror fully-synced checkpoints from a local directory to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-dir", type=Path, required=True, help="Local directory to watch.")
    parser.add_argument("--repo-id", type=str, required=True, help="Destination Hugging Face repo id.")
    parser.add_argument("--repo-type", type=str, default="model", help="Hugging Face repo type.")
    parser.add_argument("--path-prefix", type=str, default="", help="Optional prefix inside the repo.")
    parser.add_argument("--pattern", action="append", default=["epoch_*.pt"], help="Glob pattern(s) to mirror.")
    parser.add_argument("--extra-file", action="append", default=[], help="Additional file name(s) to mirror.")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval.")
    parser.add_argument("--stable-polls", type=int, default=2, help="Required identical observations before upload.")
    parser.add_argument("--token-env", type=str, default="HF_TOKEN", help="Env var holding the Hugging Face token.")
    parser.add_argument("--state-file", type=Path, default=None, help="Optional JSON state file path.")
    return parser.parse_args()


def _load_state(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _save_state(path: Path, state: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _iter_candidates(watch_dir: Path, patterns: Iterable[str], extra_files: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(watch_dir.glob(pattern)))
    for name in extra_files:
        path = watch_dir / name
        if path.exists():
            paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _record_signature(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _path_in_repo(path: Path, prefix: str) -> str:
    parts = [prefix.strip("/")] if prefix.strip("/") else []
    parts.append(path.name)
    return "/".join(parts)


def main() -> int:
    args = _parse_args()
    token = os.getenv(args.token_env)
    if not token:
        raise SystemExit(f"Missing Hugging Face token in ${args.token_env}")

    watch_dir = args.watch_dir.resolve()
    state_file = args.state_file or (watch_dir / ".hf_mirror_state.json")
    state = _load_state(state_file)
    observed: Dict[str, Dict[str, object]] = {}

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)

    print(f"[hf-mirror] watching {watch_dir}", flush=True)
    print(f"[hf-mirror] repo={args.repo_id} prefix={args.path_prefix or '.'}", flush=True)

    while True:
        for path in _iter_candidates(watch_dir, args.pattern, args.extra_file):
            sig = _record_signature(path)
            key = str(path.resolve())
            prev_seen = observed.get(key)
            if prev_seen and prev_seen.get("size") == sig["size"] and prev_seen.get("mtime_ns") == sig["mtime_ns"]:
                stable_count = int(prev_seen.get("stable_count", 1)) + 1
            else:
                stable_count = 1
            observed[key] = {
                **sig,
                "stable_count": stable_count,
            }

            uploaded = state.get(key)
            if uploaded and uploaded.get("size") == sig["size"] and uploaded.get("mtime_ns") == sig["mtime_ns"]:
                continue
            if stable_count < args.stable_polls:
                continue

            repo_path = _path_in_repo(path, args.path_prefix)
            print(f"[hf-mirror] uploading {path.name} -> {repo_path}", flush=True)
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=repo_path,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                commit_message=f"Mirror {path.name}",
            )
            state[key] = {
                "size": sig["size"],
                "mtime_ns": sig["mtime_ns"],
                "repo_path": repo_path,
                "uploaded_at": int(time.time()),
            }
            _save_state(state_file, state)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
