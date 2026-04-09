#!/usr/bin/env python3
"""
Sweep per-GPU batch sizes and report throughput + 50-epoch ETA.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


RESULT_RE = re.compile(
    r"\[result\]\s+status=(?P<status>\w+)\s+batch_size=(?P<batch_size>\d+)\s+"
    r"world_size=(?P<world_size>\d+)\s+global_batch=(?P<global_batch>\d+)"
    r"(?:\s+tok_s=(?P<tok_s>[0-9.]+))?"
    r"(?:\s+elapsed_s=(?P<elapsed_s>[0-9.]+))?"
    r"(?:\s+measured_steps=(?P<measured_steps>\d+))?"
    r"(?:\s+mean_loss=(?P<mean_loss>[0-9.]+))?"
    r"(?:\s+max_mem_gib=(?P<max_mem_gib>[0-9.]+))?"
)


@dataclass
class SweepResult:
    status: str
    batch_size: int
    world_size: int
    global_batch: int
    tok_s: float = 0.0
    elapsed_s: float = 0.0
    measured_steps: int = 0
    mean_loss: float = 0.0
    max_mem_gib: float = 0.0
    returncode: int = 0
    stderr_tail: str = ""


def _parse_batch_sizes(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        values.append(int(p))
    if not values:
        raise ValueError("No batch sizes were provided.")
    return values


def _parse_result(stdout: str, returncode: int, batch_size: int, nproc: int, stderr: str) -> SweepResult:
    matches = RESULT_RE.findall(stdout)
    if not matches:
        tail = "\n".join((stderr or "").strip().splitlines()[-8:])
        return SweepResult(
            status="error",
            batch_size=batch_size,
            world_size=nproc,
            global_batch=batch_size * nproc,
            returncode=returncode,
            stderr_tail=tail,
        )

    groups = RESULT_RE.search([m.group(0) for m in RESULT_RE.finditer(stdout)][-1])
    assert groups is not None
    d = groups.groupdict()

    return SweepResult(
        status=d["status"],
        batch_size=int(d["batch_size"]),
        world_size=int(d["world_size"]),
        global_batch=int(d["global_batch"]),
        tok_s=float(d["tok_s"] or 0.0),
        elapsed_s=float(d["elapsed_s"] or 0.0),
        measured_steps=int(d["measured_steps"] or 0),
        mean_loss=float(d["mean_loss"] or 0.0),
        max_mem_gib=float(d["max_mem_gib"] or 0.0),
        returncode=returncode,
        stderr_tail="\n".join((stderr or "").strip().splitlines()[-8:]),
    )


def _run_once(
    config: str,
    batch_size: int,
    warmup_steps: int,
    steps: int,
    nproc_per_node: int,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    num_workers: Optional[int],
    disable_compile: bool,
) -> SweepResult:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(nnodes),
        "--node_rank",
        str(node_rank),
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_addr",
        str(master_addr),
        "--master_port",
        str(master_port),
        "training/benchmark_throughput.py",
        "--config",
        config,
        "--batch-size",
        str(batch_size),
        "--warmup-steps",
        str(warmup_steps),
        "--steps",
        str(steps),
    ]
    if num_workers is not None:
        cmd.extend(["--num-workers", str(num_workers)])
    if disable_compile:
        cmd.append("--disable-compile")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined_stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return _parse_result(
        stdout=combined_stdout,
        returncode=proc.returncode,
        batch_size=batch_size,
        nproc=nproc_per_node,
        stderr=proc.stderr or "",
    )


def _format_eta_hours(hours: float) -> str:
    if hours >= 1.0:
        return f"{hours:.2f}h"
    return f"{hours * 60.0:.1f}m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-size throughput sweep (DDP)")
    parser.add_argument("--config", type=str, default="configs/real_config_8gpu_100m.yaml")
    parser.add_argument("--batch-sizes", type=str, default="24,32,40,48,56,64,72,80,96")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--stop-on-oom", dest="stop_on_oom", action="store_true")
    parser.add_argument("--no-stop-on-oom", dest="stop_on_oom", action="store_false")
    parser.add_argument("--tokens-per-epoch", type=float, default=30342999.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-json", type=str, default="sweep_results_8gpu.json")
    parser.set_defaults(stop_on_oom=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    results: List[SweepResult] = []

    print(f"[sweep] config={config_path}")
    print(f"[sweep] batch_sizes={batch_sizes}")
    print(
        "[sweep] launch "
        f"nnodes={args.nnodes} node_rank={args.node_rank} nproc_per_node={args.nproc_per_node} "
        f"master={args.master_addr}:{args.master_port}"
    )
    print(f"[sweep] warmup_steps={args.warmup_steps} measured_steps={args.steps}")

    for idx, batch_size in enumerate(batch_sizes, start=1):
        print(f"[sweep] ({idx}/{len(batch_sizes)}) batch_size={batch_size} ...")
        result = _run_once(
            config=str(config_path),
            batch_size=batch_size,
            warmup_steps=int(args.warmup_steps),
            steps=int(args.steps),
            nproc_per_node=int(args.nproc_per_node),
            nnodes=int(args.nnodes),
            node_rank=int(args.node_rank),
            master_addr=str(args.master_addr),
            master_port=int(args.master_port),
            num_workers=args.num_workers,
            disable_compile=bool(args.disable_compile),
        )
        results.append(result)

        if result.status == "ok":
            eta_hours = (args.tokens_per_epoch * args.epochs) / max(result.tok_s, 1e-9) / 3600.0
            print(
                "[sweep] ok "
                f"global_batch={result.global_batch} tok_s={result.tok_s:.1f} "
                f"max_mem_gib={result.max_mem_gib:.2f} eta_{args.epochs}ep={_format_eta_hours(eta_hours)}"
            )
        elif result.status == "oom":
            print(f"[sweep] oom at batch_size={batch_size} (global_batch={result.global_batch})")
            if args.stop_on_oom:
                break
        else:
            print(
                "[sweep] error "
                f"batch_size={batch_size} returncode={result.returncode} "
                f"stderr_tail={result.stderr_tail!r}"
            )

    ok_results = [r for r in results if r.status == "ok"]
    best = max(ok_results, key=lambda r: r.tok_s) if ok_results else None

    print("\n[sweep] summary")
    for r in results:
        if r.status == "ok":
            eta_hours = (args.tokens_per_epoch * args.epochs) / max(r.tok_s, 1e-9) / 3600.0
            print(
                f"  batch={r.batch_size:>4} global_batch={r.global_batch:>5} "
                f"tok_s={r.tok_s:>10.1f} mem_gib={r.max_mem_gib:>7.2f} "
                f"eta_{args.epochs}ep={_format_eta_hours(eta_hours)}"
            )
        else:
            print(
                f"  batch={r.batch_size:>4} global_batch={r.global_batch:>5} "
                f"status={r.status} returncode={r.returncode}"
            )

    if best is not None:
        best_eta_hours = (args.tokens_per_epoch * args.epochs) / max(best.tok_s, 1e-9) / 3600.0
        print("\n[sweep] best")
        print(
            f"  batch_size={best.batch_size} global_batch={best.global_batch} "
            f"tok_s={best.tok_s:.1f} max_mem_gib={best.max_mem_gib:.2f} "
            f"eta_{args.epochs}ep={_format_eta_hours(best_eta_hours)}"
        )
    else:
        print("\n[sweep] no successful runs")

    save_path = Path(args.save_json)
    payload = {
        "config": str(config_path),
        "epochs": int(args.epochs),
        "tokens_per_epoch": float(args.tokens_per_epoch),
        "results": [asdict(r) for r in results],
        "best": asdict(best) if best else None,
        "best_eta_hours": (
            (args.tokens_per_epoch * args.epochs) / max(best.tok_s, 1e-9) / 3600.0 if best else None
        ),
    }
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {save_path}")

    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
