#!/usr/bin/env python3
"""
Launch a matched AR-vs-diffusion 100M comparison on one multi-GPU node.

The script:
- materializes run-specific configs
- runs a throughput micro-benchmark for each model
- trains AR and diffusion sequentially on the same hardware
- runs final chemistry-aware eval from the last checkpoint
- writes a compact JSON and Markdown summary
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _stream_command(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}): {' '.join(cmd)}")


def _extract_result_json(log_path: Path) -> Dict:
    marker = "[result_json] "
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise RuntimeError(f"Missing [result_json] payload in {log_path}")


def _latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = []
    for path in output_dir.glob("epoch_*.pt"):
        match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def _run_chemical_eval(
    mode: str,
    config_path: Path,
    checkpoint_path: Path,
    data_dir: str,
    split: str,
    max_examples: int,
    output_json: Path,
) -> Dict:
    cmd = [
        sys.executable,
        "scripts/eval_chemical_metrics.py",
        "--checkpoint",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--data_dir",
        data_dir,
        "--split",
        split,
        "--max_examples",
        str(max_examples),
        "--model_type",
        mode,
        "--output_json",
        str(output_json),
    ]
    _stream_command(cmd, ROOT, output_json.with_suffix(".log"))
    with output_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _materialize_config(
    base_cfg: Dict,
    output_dir: Path,
    batch_size: Optional[int],
    num_epochs: Optional[int],
    learning_rate: Optional[float],
) -> Dict:
    cfg = deepcopy(base_cfg)
    if batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = int(batch_size)
        cfg["training"]["test_batch_size"] = int(batch_size)
    if num_epochs is not None:
        cfg.setdefault("training", {})["num_epochs"] = int(num_epochs)
    if learning_rate is not None:
        cfg.setdefault("training", {})["learning_rate"] = float(learning_rate)
    cfg.setdefault("checkpoint", {})["output_dir"] = str(output_dir)
    return cfg


def _summary_markdown(summary: Dict) -> str:
    lines = [
        f"# 100M Modality Comparison - {summary['run_name']}",
        "",
        "## Setup",
        f"- GPUs: `{summary['nproc_per_node']}`",
        f"- Base batch size per GPU: `{summary['batch_size']}`",
        f"- Epochs per run: `{summary['num_epochs']}`",
        f"- Data: `{summary['data_dir']}`",
        f"- Run root: `{summary['run_root']}`",
        "",
        "## Throughput",
    ]
    for mode in ["ar", "diffusion"]:
        bench = summary[mode]["throughput_benchmark"]
        lines.append(
            f"- {mode}: tok/s `{bench['tok_s']}`, step_ms `{bench['step_ms']}`, max_mem_gib `{bench['max_mem_gib']}`"
        )
    lines.extend(["", "## Training",])
    for mode in ["ar", "diffusion"]:
        train = summary[mode]["training"]
        latest = train.get("latest_metrics", {})
        lines.append(
            f"- {mode}: wallclock_s `{train['wallclock_s']:.1f}`, epoch `{latest.get('epoch')}`, "
            f"train_loss `{latest.get('train_loss')}`, val_loss `{latest.get('val_loss')}`, tok_s `{latest.get('tok_s')}`"
        )
        if "decode_valid_smiles" in latest:
            lines.append(
                f"  decode_valid `{latest.get('decode_valid_smiles')}`, decode_exact `{latest.get('decode_exact_match_all')}`, "
                f"tanimoto `{latest.get('decode_avg_tanimoto')}`, ecfp6_iou `{latest.get('decode_avg_ecfp6_iou')}`, mcs `{latest.get('decode_avg_mcs_over_target')}`"
            )
    lines.extend(["", "## Final Chemical Eval",])
    for mode in ["ar", "diffusion"]:
        for split, payload in summary[mode]["chemical_eval"].items():
            lines.append(
                f"- {mode}/{split}: valid `{payload['valid_smiles']}`, exact `{payload['exact_match_all']}`, "
                f"tanimoto `{payload['avg_tanimoto']}`, ecfp6_iou `{payload['avg_ecfp6_iou']}`, mcs `{payload['avg_mcs_over_target']}`"
            )
    return "\n".join(lines) + "\n"


def _dist_launch_prefix(nproc_per_node: int) -> List[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matched 100M AR-vs-diffusion comparison")
    parser.add_argument("--config-ar", type=str, default="configs/compare_h100_100m_ar.yaml")
    parser.add_argument("--config-diffusion", type=str, default="configs/compare_h100_100m_diffusion.yaml")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--ar-learning-rate", type=float, default=None)
    parser.add_argument("--diffusion-learning-rate", type=float, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=80)
    parser.add_argument("--benchmark-warmup-steps", type=int, default=20)
    parser.add_argument("--chemical-splits", type=str, default="test")
    parser.add_argument("--chemical-max-examples", type=int, default=256)
    parser.add_argument("--run-root", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"modality_compare_100m_{timestamp}"
    run_root = Path(args.run_root) if args.run_root else ROOT / "artifacts" / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    base_ar = _load_yaml(ROOT / args.config_ar)
    base_diff = _load_yaml(ROOT / args.config_diffusion)
    batch_size = int(args.batch_size or base_ar["training"]["batch_size"])
    num_epochs = int(args.num_epochs or base_ar["training"]["num_epochs"])
    data_dir = str(base_ar["data"]["tokenized_dir"])

    ar_output_dir = run_root / "ar_checkpoints"
    diff_output_dir = run_root / "diffusion_checkpoints"

    ar_cfg = _materialize_config(base_ar, ar_output_dir, batch_size, num_epochs, args.ar_learning_rate)
    diff_cfg = _materialize_config(base_diff, diff_output_dir, batch_size, num_epochs, args.diffusion_learning_rate)

    ar_cfg_path = run_root / "configs" / "ar.yaml"
    diff_cfg_path = run_root / "configs" / "diffusion.yaml"
    _write_yaml(ar_cfg_path, ar_cfg)
    _write_yaml(diff_cfg_path, diff_cfg)

    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_root": str(run_root),
        "nproc_per_node": int(args.nproc_per_node),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "data_dir": data_dir,
        "ar": {},
        "diffusion": {},
    }

    for mode, cfg_path, output_dir in [
        ("ar", ar_cfg_path, ar_output_dir),
        ("diffusion", diff_cfg_path, diff_output_dir),
    ]:
        bench_log = run_root / f"benchmark_{mode}.log"
        bench_cmd = _dist_launch_prefix(args.nproc_per_node) + [
            "training/benchmark_throughput.py",
            "--mode",
            mode,
            "--config",
            str(cfg_path),
            "--batch-size",
            str(batch_size),
            "--steps",
            str(args.benchmark_steps),
            "--warmup-steps",
            str(args.benchmark_warmup_steps),
        ]
        _stream_command(bench_cmd, ROOT, bench_log)
        throughput = _extract_result_json(bench_log)

        train_log = run_root / f"train_{mode}.log"
        train_script = "training/train_autoregressive.py" if mode == "ar" else "training/train_diffusion.py"
        train_cmd = _dist_launch_prefix(args.nproc_per_node) + [
            train_script,
            "--config",
            str(cfg_path),
        ]
        started = time.perf_counter()
        started_at_utc = datetime.now(timezone.utc).isoformat()
        _stream_command(train_cmd, ROOT, train_log)
        wallclock_s = time.perf_counter() - started
        ended_at_utc = datetime.now(timezone.utc).isoformat()

        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        checkpoint_path = _latest_checkpoint(output_dir)

        chemical_eval: Dict[str, Dict] = {}
        for split in [part.strip() for part in args.chemical_splits.split(",") if part.strip()]:
            eval_json = run_root / f"chemical_eval_{mode}_{split}.json"
            chemical_eval[split] = _run_chemical_eval(
                mode=mode,
                config_path=cfg_path,
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                split=split,
                max_examples=int(args.chemical_max_examples),
                output_json=eval_json,
            )

        summary[mode] = {
            "config_path": str(cfg_path),
            "throughput_benchmark": throughput,
            "training": {
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "checkpoint_path": str(checkpoint_path),
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "wallclock_s": wallclock_s,
                "latest_metrics": manifest.get("latest_metrics", {}),
            },
            "chemical_eval": chemical_eval,
        }

        summary_json = run_root / "summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    summary_md = run_root / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(_summary_markdown(summary))

    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {summary_md}")


if __name__ == "__main__":
    main()
