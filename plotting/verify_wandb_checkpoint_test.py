"""Re-run test evaluation from a W&B run's saved checkpoint and compare to logged test F1.

Fetches Hydra CLI overrides from the run (same source as ``generate_rerun_sh``), composes
``train.yaml``, sets ``train=false``, ``test=true``, and ``ckpt_path`` to the path logged in
W&B summary as ``checkpoint`` (written in ``ogbench/run.py`` after training).

Run from repo root (needs data paths in overrides to resolve, same as training)::

    python plotting/verify_wandb_checkpoint_test.py
    python plotting/verify_wandb_checkpoint_test.py --run-id 0lg0er9x
    python plotting/verify_wandb_checkpoint_test.py --trainer-devices=[0]

Requires: W&B API access, checkpoint file on disk at the logged path (or override with a
local copy via ``--ckpt-path``), and the same dataset layout as the original job.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import rootutils
import wandb
from hydra import compose, initialize_config_dir
# Repo root (parent of plotting/)
REPO_ROOT = Path(__file__).resolve().parents[1]
_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from generate_rerun_sh import (  # noqa: E402
    _OVERRIDE_KEYS_DROP,
    _override_key,
    hydra_overrides_from_wandb_run,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def _unwrap_metric(v):
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().cpu()
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:
            pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _wandb_summary_dict(run) -> dict:
    try:
        return dict(run.summary) if run.summary else {}
    except Exception:
        return {}


def _summary_test_f1_macro(summary: dict) -> float | None:
    for key in ("best_test/f1_macro", "best_test_f1_macro"):
        if key in summary and summary[key] is not None:
            try:
                return float(summary[key])
            except (TypeError, ValueError):
                continue
    return None


def _filter_inference_overrides(overrides: list[str]) -> list[str]:
    drop_keys = _OVERRIDE_KEYS_DROP | {"train", "test", "ckpt_path"}
    out: list[str] = []
    for o in overrides:
        k = _override_key(o)
        if k in drop_keys:
            continue
        out.append(o)
    return out


def _pick_run(
    api: wandb.Api,
    entity: str,
    project: str,
    *,
    run_id: str | None,
    random_seed: int,
    max_scan: int,
):
    path = f"{entity}/{project}"
    if run_id:
        return api.run(f"{path}/{run_id}")

    runs = api.runs(path, per_page=max_scan)
    candidates = []
    for run in runs:
        if getattr(run, "state", None) != "finished":
            continue
        summary = _wandb_summary_dict(run)
        if not summary.get("checkpoint"):
            continue
        if hydra_overrides_from_wandb_run(run) is None:
            continue
        if _summary_test_f1_macro(summary) is None:
            continue
        candidates.append(run)

    if not candidates:
        raise RuntimeError(
            f"No suitable finished runs in {path!r} (first {max_scan} by API order): "
            "need summary.checkpoint, Hydra args in summary/metadata, and best test F1."
        )

    rng = random.Random(random_seed)
    return rng.choice(candidates)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="bioshape-lab", help="W&B entity")
    p.add_argument("--project", default="bgbench_dataset_grid_search_final", help="W&B project")
    p.add_argument("--run-id", default=None, help="Specific run id; if omitted, pick random eligible run")
    p.add_argument("--random-seed", type=int, default=42, help="Seed when choosing a random run")
    p.add_argument("--max-scan", type=int, default=100, help="How many runs to scan when picking random")
    p.add_argument(
        "--ckpt-path",
        default=None,
        help="Override checkpoint path (must match the trained weights; default: run.summary['checkpoint'])",
    )
    p.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override (repeatable), e.g. trainer.devices=[0]",
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance when comparing F1 to W&B summary",
    )
    args = p.parse_args()

    api = wandb.Api()
    run = _pick_run(
        api,
        args.entity,
        args.project,
        run_id=args.run_id,
        random_seed=args.random_seed,
        max_scan=args.max_scan,
    )

    summary = _wandb_summary_dict(run)
    wb_f1 = _summary_test_f1_macro(summary)
    ckpt_raw = args.ckpt_path or summary.get("checkpoint")
    if not ckpt_raw:
        raise SystemExit("No checkpoint path: set --ckpt-path or ensure W&B summary contains 'checkpoint'.")

    ckpt_path = Path(str(ckpt_raw).strip()).expanduser()
    if not ckpt_path.is_file():
        raise SystemExit(
            f"Checkpoint file not found: {ckpt_path}\n"
            "Use --ckpt-path pointing to a local copy, or run on the machine where training outputs exist."
        )

    base_overrides = hydra_overrides_from_wandb_run(run)
    if not base_overrides:
        raise SystemExit("Could not read Hydra overrides from W&B (summary.args / metadata.args).")

    filtered = _filter_inference_overrides(base_overrides)
    # compose() omits cfg.hydra unless return_hydra_config=True; paths still use
    # ${hydra:runtime.output_dir} / ${hydra:runtime.cwd}, which then fail to resolve.
    # Override paths so nothing depends on HydraConfig.
    work_dir = Path.cwd().resolve()
    output_dir = Path(tempfile.mkdtemp(prefix="verify_wandb_ckpt_"))
    tail = [
        "train=false",
        "test=true",
        f"ckpt_path={ckpt_path.resolve().as_posix()}",
        "logger.wandb.offline=true",
        f"paths.output_dir={output_dir.as_posix()}",
        f"paths.work_dir={work_dir.as_posix()}",
    ]
    overrides = filtered + list(args.extra_override) + tail

    print(f"Using run: {run.id} ({getattr(run, 'name', '')})")
    print(f"Checkpoint: {ckpt_path.resolve()}")
    print(f"W&B summary best test F1 (macro): {wb_f1}")
    print("Hydra overrides (incl. train/test/ckpt):", " ".join(overrides[:8]), "..." if len(overrides) > 8 else "")

    config_dir = REPO_ROOT / "configs"
    if not config_dir.is_dir():
        raise SystemExit(f"Config dir not found: {config_dir}")

    from ogbench.run import run as ogbench_run  # noqa: PLC0415 — after rootutils

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir), job_name="verify_ckpt"):
        cfg = compose(config_name="train", overrides=overrides)

    metric_dict, _ = ogbench_run(cfg)

    local_f1 = _unwrap_metric(metric_dict.get("test/f1_macro"))
    print(f"Local test/f1_macro after ckpt load: {local_f1}")

    if wb_f1 is not None and local_f1 is not None:
        diff = abs(local_f1 - wb_f1)
        ok = diff <= args.atol
        print(f"Absolute difference vs W&B summary: {diff:g} (atol={args.atol:g})")
        print("MATCH" if ok else "MISMATCH (see note: different hardware, Lightning version, or data path)")
        return 0 if ok else 1

    print("Could not compare (missing wb_f1 or local test/f1_macro). Full metric_dict keys:")
    print(sorted(str(k) for k in metric_dict.keys()))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
