"""Build a bash script to re-train runs that have no test F1 in the exported CSV.

Uses the CSV only to list ``run_id``s (same criterion as ``load_results``). The exact Hydra
CLI is read from W&B (``summary.args`` or run ``metadata.args``), not from flattened CSV
columns, so ``${paths.log_dir}``-style overrides stay correct.

Run from repo root::

    python plotting/generate_rerun_sh.py --report
    python plotting/generate_rerun_sh.py --out plotting/rerun_missing_test.sh --n-gpus 8
    python plotting/generate_rerun_sh.py ... --format legacy --train-cmd ogbench-train  # old one-line style
    python plotting/generate_rerun_sh.py ... --progress-every 1   # log every run (noisy)

Default ``--format cluster`` writes blocks like ``python -m ogbench \\`` + indented overrides,
``trainer.devices=\\[k] \\``, and ``&`` (plus a final ``wait``), similar to multirun launch scripts.

Output ``.sh`` files use Unix LF newlines only (no CRLF), including ``newline='\\n'`` on write,
so backslash line continuation works under bash on Windows-generated paths and W&B strings
with stray ``\\r`` do not break the script.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

_PLOT_DIR = Path(__file__).resolve().parent
if str(_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOT_DIR))

import load_results as lr

# Extra W&B tag on every rerun; paths / hydra output dir come from the current machine.
RERUN_WANDB_TAG = 'rerun'
_OVERRIDE_KEYS_DROP = frozenset({'hydra.run.dir', 'paths.root_dir'})


def _override_key(override: str) -> str:
    if '=' not in override:
        return override
    return override.split('=', 1)[0].lstrip('+')


def _append_tag_to_wandb_tags_line(line: str, tag: str) -> str:
    prefix, rhs = line.split('=', 1)
    rhs_st = rhs.strip()
    if rhs_st.startswith('[') and rhs_st.endswith(']'):
        inner = rhs_st[1:-1].strip()
        parts = [p.strip() for p in inner.split(',') if p.strip()]
        if tag in parts:
            return line
        parts.append(tag)
        return f"{prefix}=[{','.join(parts)}]"
    return f'{prefix}={rhs_st},{tag}'


def prepare_rerun_overrides(
    overrides: list[str], *, rerun_tag: str = RERUN_WANDB_TAG
) -> list[str]:
    """Drop machine-specific Hydra keys and ensure ``logger.wandb.tags`` includes ``rerun_tag``."""
    filtered: list[str] = []
    for o in overrides:
        if _override_key(o) in _OVERRIDE_KEYS_DROP:
            continue
        filtered.append(o)

    out: list[str] = []
    tags_done = False
    for o in filtered:
        if _override_key(o) == 'logger.wandb.tags':
            tags_done = True
            out.append(_append_tag_to_wandb_tags_line(o, rerun_tag))
        else:
            out.append(o)
    if not tags_done:
        out.append(f'logger.wandb.tags=[{rerun_tag}]')
    return out


def _bash_script_unix_text(lines: list[str]) -> str:
    """LF-only script body: bash ``\\`` continuation requires no ``\\r`` before newline."""
    normalized: list[str] = []
    for line in lines:
        s = str(line).replace('\r\n', '\n').replace('\r', '')
        normalized.append(s)
    return '\n'.join(normalized) + '\n'


def _wandb_summary_dict(run) -> dict:
    try:
        return dict(run.summary) if run.summary else {}
    except Exception:
        return {}


def _wandb_run_metadata_dict(run) -> dict:
    m = getattr(run, 'metadata', None)
    if isinstance(m, dict):
        return m
    attrs = getattr(run, '_attrs', None) or {}
    m = attrs.get('metadata')
    return m if isinstance(m, dict) else {}


def hydra_overrides_from_wandb_run(run) -> list[str] | None:
    """Hydra CLI overrides as logged by the job (W&B ``summary.args`` or run metadata ``args``)."""
    raw: list | None = None
    summary = _wandb_summary_dict(run)
    if isinstance(summary.get('args'), (list, tuple)):
        raw = [x for x in summary['args'] if x is not None]
    if not raw:
        meta = _wandb_run_metadata_dict(run)
        if isinstance(meta.get('args'), (list, tuple)):
            raw = [x for x in meta['args'] if x is not None]
    if not raw:
        return None
    out: list[str] = []
    for x in raw:
        if isinstance(x, (list, dict)):
            continue
        s = str(x).strip()
        if not s:
            continue
        out.append(s)
    i = 0
    while i < len(out) and '=' not in out[i] and not out[i].startswith('+'):
        i += 1
    out = out[i:]
    return out if out else None


def _shell_token_for_hydra_override(s: str) -> str:
    """Quote for bash; avoid single quotes around tokens that need ``${...}`` expansion."""
    if '${' in s:
        unsafe = frozenset(' \t\n;&|()<>')
        if any(c in s for c in unsafe):
            return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
        return s
    return shlex.quote(s)


def _continuation_line_token(s: str) -> str:
    """Hydra token for a continued shell line: bare when safe, else quoted (keeps ``${...}`` unquoted)."""
    if '${' in s:
        unsafe = frozenset(' \t\n;&|()<>')
        if any(c in s for c in unsafe):
            return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
        return s
    if any(c in s for c in ' \t\n"\'`;$\\!*?()<>|&'):
        return shlex.quote(s)
    return s


def _overrides_without_trainer_devices(overrides: list[str]) -> list[str]:
    return [o for o in overrides if not str(o).startswith('trainer.devices=')]


def format_cluster_rerun_block(
    overrides: list[str],
    launcher_parts: list[str],
    gpu_slot: int,
    *,
    n_gpus: int,
    background: bool,
) -> str:
    """One ``python -m ogbench \\`` … ``trainer.devices=\\[i]`` … ``&`` block (multiline)."""
    ov = _overrides_without_trainer_devices(overrides)
    body: list[str] = [' '.join(launcher_parts) + ' \\']
    for o in ov:
        body.append('    ' + _continuation_line_token(o) + ' \\')
    if n_gpus and n_gpus > 0:
        body.append(f'    trainer.devices=\\[{gpu_slot}\\] \\')
    if background:
        body.append('    &')
    elif body:
        last = body[-1].rstrip()
        if last.endswith('\\'):
            last = last[:-1].rstrip()
        body[-1] = last
    return '\n'.join(body)


def _should_log_progress(idx: int, total: int, every: int) -> bool:
    if every <= 1 or total <= 1:
        return True
    if idx == 0 or idx == total - 1:
        return True
    return (idx + 1) % every == 0


def write_rerun_shell_for_missing_test_f1(
    run_ids: list[str],
    out_sh_path: str | os.PathLike,
    *,
    wandb_entity: str,
    wandb_project: str,
    train_cmd: str = 'python -m ogbench',
    n_gpus: int = 1,
    api_timeout: int = 120,
    max_retries: int = 4,
    progress_every: int = 5,
    output_format: str = 'cluster',
    background: bool = True,
    append_wait: bool = True,
    rerun_tag: str = RERUN_WANDB_TAG,
) -> tuple[int, int]:
    """Fetch each run from W&B, read Hydra ``args``, write one rerun block per run.

    ``output_format``: ``cluster`` (multiline ``python -m ogbench`` + ``trainer.devices``) or
    ``legacy`` (single line + ``CUDA_VISIBLE_DEVICES``).

    Returns ``(n_written, n_skipped)``.
    """
    out_path = Path(out_sh_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(run_ids)
    fmt = output_format.lower().strip()
    if fmt not in ('cluster', 'legacy'):
        raise ValueError(f"output_format must be 'cluster' or 'legacy', got {output_format!r}")

    print(f'Initializing W&B API (timeout={api_timeout}s per request)…', flush=True)
    t_api = time.perf_counter()
    api = wandb.Api(timeout=api_timeout)
    print(
        f'W&B API ready in {time.perf_counter() - t_api:.1f}s. Fetching {total} run(s)…',
        flush=True,
    )
    t_all = time.perf_counter()

    written = 0
    skipped: list[tuple[str, str]] = []

    if fmt == 'cluster':
        launcher_parts = shlex.split(train_cmd, posix=True)
        if not launcher_parts:
            raise ValueError(
                'cluster format needs a non-empty --train-cmd (e.g. python -m ogbench)'
            )
        lines = [
            '#!/bin/bash',
            '# Re-run training: rows missing best_test_f1_macro in the export CSV.',
            '# Overrides from W&B (summary or metadata ``args``).',
            f'# format=cluster  n_gpus={n_gpus}  launcher={train_cmd!r}  background={background}',
            '# Drops: hydra.run.dir, paths.root_dir. Adds W&B tag: ' f'{rerun_tag!r}',
            '',
        ]
    else:
        lines = [
            '#!/usr/bin/env bash',
            '# Re-run training (legacy one-line + CUDA_VISIBLE_DEVICES).',
            f'# format=legacy  n_gpus={n_gpus}  train_cmd={train_cmd!r}',
            '# Drops: hydra.run.dir, paths.root_dir. W&B tag: ' f'{rerun_tag!r}',
            'set -euo pipefail',
            '',
        ]

    for idx, run_id in enumerate(run_ids):
        t0 = time.perf_counter()
        if _should_log_progress(idx, total, progress_every):
            print(f'  [{idx + 1}/{total}] fetching {run_id!r} …', flush=True)

        run = None
        last_err = ''
        for attempt in range(max_retries):
            try:
                run = api.run(f'{wandb_entity}/{wandb_project}/{run_id}')
                break
            except Exception as e:
                last_err = str(e)
                if lr._is_retryable_error(e) and attempt < max_retries - 1:
                    if _should_log_progress(idx, total, progress_every):
                        print(
                            f'      retry {attempt + 1}/{max_retries} after error: {last_err[:120]}',
                            flush=True,
                        )
                    time.sleep(min(60.0, (2**attempt) * 2 + float(np.random.uniform(0, 2))))
                else:
                    break
        dt = time.perf_counter() - t0

        if run is None:
            skipped.append((run_id, last_err or 'fetch failed'))
            lines.append(f'# SKIP run_id={run_id} (could not load run: {last_err})')
            if _should_log_progress(idx, total, progress_every):
                err_short = (last_err or 'fetch failed').replace('\n', ' ')[:100]
                print(f'      SKIP fetch failed ({dt:.1f}s): {err_short}', flush=True)
            continue

        overrides = hydra_overrides_from_wandb_run(run)
        if not overrides:
            skipped.append((run_id, 'no args in summary/metadata'))
            lines.append(f'# SKIP run_id={run_id} (no Hydra args list in W&B summary or metadata)')
            if _should_log_progress(idx, total, progress_every):
                print(f'      SKIP no Hydra args in W&B summary/metadata ({dt:.1f}s)', flush=True)
            continue

        overrides = prepare_rerun_overrides(overrides, rerun_tag=rerun_tag)

        gpu_slot = idx % n_gpus if n_gpus else 0
        if fmt == 'cluster':
            if written > 0:
                lines.append('')
            block = format_cluster_rerun_block(
                overrides,
                launcher_parts,
                gpu_slot,
                n_gpus=n_gpus,
                background=background,
            )
            lines.append(block)
            written += 1
        else:
            train_tok = shlex.quote(train_cmd) if train_cmd else 'ogbench-train'
            body = (
                train_tok + ' ' + ' '.join(_shell_token_for_hydra_override(o) for o in overrides)
            )
            if n_gpus and n_gpus > 0:
                line = f'CUDA_VISIBLE_DEVICES={gpu_slot} {body}'
            else:
                line = body
            lines.append(line)
            written += 1
        if _should_log_progress(idx, total, progress_every):
            if fmt == 'cluster' and n_gpus:
                gpu_info = f'trainer.devices=[{gpu_slot}]'
            elif fmt == 'cluster':
                gpu_info = 'no device pin'
            else:
                gpu_info = f'CUDA_VISIBLE_DEVICES={gpu_slot}' if n_gpus else '-'
            print(
                f'      ok  {len(overrides)} override(s)  {gpu_info}  ({dt:.1f}s)',
                flush=True,
            )

    if fmt == 'cluster' and background and append_wait and written:
        lines.append('')
        lines.append('wait')

    lines.append('')
    if skipped:
        lines.append('# --- skipped (inspect in W&B UI or re-export) ---')
        for rid, reason in skipped:
            lines.append(f'# {rid}\t{reason}')

    print(
        f'Finished W&B passes in {time.perf_counter() - t_all:.1f}s '
        f'({written} command line(s), {len(skipped)} skipped). Writing {out_path}…',
        flush=True,
    )
    out_path.write_text(
        _bash_script_unix_text(lines),
        encoding='utf-8',
        newline='\n',
    )
    try:
        os.chmod(out_path, 0o755)
    except OSError:
        pass
    return written, len(skipped)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--csv',
        default=None,
        help=f'Exported runs CSV (default: {lr.csv_filename})',
    )
    p.add_argument(
        '--report',
        action='store_true',
        help='Print how many rows lack best_test_f1_macro and exit.',
    )
    p.add_argument(
        '--out',
        metavar='PATH.sh',
        default=None,
        help='Write bash script here (required unless --report).',
    )
    p.add_argument(
        '--train-cmd',
        default='python -m ogbench',
        help='Launcher split for cluster format (default: python -m ogbench). Legacy format: e.g. ogbench-train.',
    )
    p.add_argument(
        '--format',
        choices=('cluster', 'legacy'),
        default='cluster',
        help='cluster: multiline + trainer.devices=[k] + & (default). legacy: one line + CUDA_VISIBLE_DEVICES.',
    )
    p.add_argument(
        '--no-background',
        action='store_true',
        help='Cluster format only: omit trailing & (sequential-friendly).',
    )
    p.add_argument(
        '--no-wait',
        action='store_true',
        help='Cluster format only: do not append ``wait`` after background jobs.',
    )
    p.add_argument(
        '--rerun-tag',
        default=RERUN_WANDB_TAG,
        metavar='NAME',
        help=f'Appended to logger.wandb.tags (default: {RERUN_WANDB_TAG!r}).',
    )
    p.add_argument(
        '--n-gpus',
        type=int,
        default=1,
        metavar='N',
        help='cluster: trainer.devices=[i %% N]. legacy: CUDA_VISIBLE_DEVICES=i%%N. Use 0 to omit pinning.',
    )
    p.add_argument('--wandb-entity', default=None, help=f'Default: {lr.wandb_username}')
    p.add_argument('--wandb-project', default=None, help=f'Default: {lr.wandb_project}')
    p.add_argument(
        '--progress-every',
        type=int,
        default=5,
        metavar='N',
        help='Log each N-th run plus first and last (default: 5). Use 1 to log every run.',
    )
    args = p.parse_args()

    csv_path = args.csv or lr.csv_filename
    entity = args.wandb_entity or lr.wandb_username
    project = args.wandb_project or lr.wandb_project

    df = pd.read_csv(csv_path, low_memory=False)
    m = lr.runs_missing_best_test_f1_macro_mask(df)

    if args.report:
        n = int(m.sum())
        print(f'CSV: {csv_path}')
        print(f'Rows missing {lr.MISSING_TEST_F1_COL}: {n:,} / {len(df):,}')
        if n and 'state' in df.columns:
            print('state breakdown (missing test F1):')
            print(df.loc[m, 'state'].value_counts(dropna=False).to_string())
        return

    if not args.out:
        p.error('--out PATH.sh is required (unless using --report)')

    run_ids = df.loc[m, 'run_id'].astype(str).tolist()
    print(f'CSV: {csv_path}')
    print(f'Runs missing {lr.MISSING_TEST_F1_COL}: {len(run_ids):,}')
    print(f'Fetching Hydra args from W&B ({entity}/{project})…', flush=True)
    n_ok, n_skip = write_rerun_shell_for_missing_test_f1(
        run_ids,
        args.out,
        wandb_entity=entity,
        wandb_project=project,
        train_cmd=args.train_cmd,
        n_gpus=args.n_gpus,
        api_timeout=120,
        progress_every=max(1, args.progress_every),
        output_format=args.format,
        background=not args.no_background,
        append_wait=not args.no_wait,
        rerun_tag=args.rerun_tag,
    )
    print(
        f'Wrote {args.out!r}: {n_ok} command(s), {n_skip} skipped (see comments at end of script).'
    )


if __name__ == '__main__':
    main()
