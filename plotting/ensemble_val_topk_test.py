"""CLI wrapper for ``ensemble_val_topk_core.run_ensemble`` (single model + dataset).

See ``ensemble_val_topk_core`` module docstring and ``export_ensemble_topk_aggregated.py``
for batch CSV export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rootutils

REPO_ROOT = Path(__file__).resolve().parents[1]
_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

from ensemble_val_topk_core import (  # noqa: E402
    EnsembleRunConfig,
    print_cli_summary,
    run_ensemble,
)

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--csv', default=str(REPO_ROOT / 'plotting/final_results_hyperparams_neurips.csv')
    )
    p.add_argument('--entity', default='bioshape-lab')
    p.add_argument('--project', default='bgbench_dataset_grid_search_final')
    p.add_argument('--model', required=True, help='Canonical model name, e.g. gps, gin')
    p.add_argument('--dataset', required=True, help='data_name, e.g. addneuromed, motrpac')
    p.add_argument('--min-seeds-per-bucket', type=int, default=1)
    p.add_argument('--max-buckets', type=int, default=5)
    p.add_argument('--ks', default='1,3,5')
    p.add_argument(
        '--graph-lock',
        action='store_true',
        help="Restrict to best bucket's graph slice + shared test loader (old behaviour).",
    )
    p.add_argument('--wandb-offline', action='store_true')
    p.add_argument('--extra-override', action='append', default=[])
    args = p.parse_args()

    ks = tuple(int(x.strip()) for x in args.ks.split(',') if x.strip())
    if not ks or any(k < 1 for k in ks):
        raise SystemExit('--ks must list positive integers.')
    if min(ks) > args.max_buckets:
        raise SystemExit(f'min ks ({min(ks)}) exceeds --max-buckets ({args.max_buckets}).')

    cfg = EnsembleRunConfig(
        csv_path=Path(args.csv),
        entity=args.entity,
        project=args.project,
        model=args.model,
        dataset=args.dataset,
        ks=ks,
        min_seeds_per_bucket=int(args.min_seeds_per_bucket),
        max_buckets=int(args.max_buckets),
        wandb_offline=bool(args.wandb_offline),
        extra_override=tuple(args.extra_override or ()),
        work_dir=Path.cwd().resolve(),
        config_dir=REPO_ROOT / 'configs',
        graph_lock=bool(args.graph_lock),
        verbose=True,
    )
    rows, msg = run_ensemble(cfg)
    if not rows:
        print(msg, flush=True)
        return 1
    print(msg, flush=True)
    print_cli_summary(cfg, rows, ks)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
