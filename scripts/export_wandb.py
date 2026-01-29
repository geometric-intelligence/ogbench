from pathlib import Path
from typing import Any

import pandas as pd
import wandb


def flatten_dict(d: dict[str, Any], parent_key: str = '', sep: str = '_') -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main() -> None:
    api = wandb.Api()
    runs = api.runs('<anonymous>/biggraphbench')

    summary_list: list[dict[str, Any]] = []
    config_list: list[dict[str, Any]] = []
    name_list: list[str] = []

    for run in runs:
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})
        name_list.append(run.name)

    # Flatten the nested dictionaries
    flat_summaries = [flatten_dict(s) for s in summary_list]
    flat_configs = [flatten_dict(c) for c in config_list]

    # Create DataFrames
    summary_df = pd.DataFrame(flat_summaries)
    config_df = pd.DataFrame(flat_configs)
    name_df = pd.DataFrame({'name': name_list})

    # Combine all DataFrames
    final_df = pd.concat([name_df, config_df, summary_df], axis=1)

    # Save to CSV
    output_path = Path('wandb_export.csv')
    final_df.to_csv(output_path, index=False)
    print(f'Data exported to {output_path}')


if __name__ == '__main__':
    main()
