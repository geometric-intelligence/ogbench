#!/usr/bin/env python3
"""Download GenePT gene embedding pickles from Zenodo into data/genept/.

Source: https://zenodo.org/records/10833191
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_ZIP_URL = 'https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1'
DEFAULT_FILES = (
    'GenePT_gene_embedding_ada_text.pickle',
    'GenePT_gene_protein_embedding_model_3_text.pickle',
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('data/genept'),
        help='Directory to extract GenePT pickles into',
    )
    parser.add_argument(
        '--url',
        default=ZENODO_ZIP_URL,
        help='Zenodo zip URL',
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = args.out_dir / 'GenePT_emebdding_v2.zip'
    if not zip_path.exists():
        print(f'Downloading {args.url}')
        urlretrieve(args.url, zip_path)  # nosec B310
    else:
        print(f'Using existing {zip_path}')

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        print(f'Archive contains {len(names)} entries')
        for name in names:
            base = Path(name).name
            if base in DEFAULT_FILES or base.endswith('.pickle'):
                target = args.out_dir / base
                if target.exists():
                    print(f'Skip existing {target}')
                    continue
                print(f'Extracting {base} -> {target}')
                with zf.open(name) as src, open(target, 'wb') as dst:
                    dst.write(src.read())

    print('Done. Point gene_identity.embeddings_path at one of:')
    for f in sorted(args.out_dir.glob('*.pickle')):
        print(f'  {f}')


if __name__ == '__main__':
    main()
