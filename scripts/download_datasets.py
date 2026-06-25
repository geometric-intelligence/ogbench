#!/usr/bin/env python3
"""CLI script for downloading and uploading datasets to HuggingFace."""


import shutil

import typer

from scripts.processors.addneuromed import process_addneuromed
from scripts.processors.brca import process_brca
from scripts.processors.motrpac import process_motrpac
from scripts.processors.parkinsons import process_parkinsons
from scripts.processors.smoking import process_smoking
from scripts.processors.tuberculosis import process_tuberculosis

app = typer.Typer(help='Download and upload omics datasets to HuggingFace')


@app.command()
def motrpac(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload MotrPac dataset to HuggingFace."""
    typer.echo('Processing MotrPac dataset...')
    process_motrpac(output_dir)
    typer.echo('✅ MotrPac dataset processed and uploaded successfully!')


@app.command()
def addneuromed(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload AddNeuroMed dataset to HuggingFace."""
    typer.echo('Processing AddNeuroMed dataset...')
    process_addneuromed(output_dir)
    typer.echo('✅ AddNeuroMed dataset processed and uploaded successfully!')


@app.command()
def parkinsons(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload Parkinsons dataset to HuggingFace."""
    typer.echo('Processing Parkinsons dataset...')
    process_parkinsons(output_dir)
    typer.echo('✅ Parkinsons dataset processed and uploaded successfully!')


@app.command()
def smoking(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload Smoking (GSE50660) dataset to HuggingFace."""
    typer.echo('Processing Smoking dataset...')
    process_smoking(output_dir)
    typer.echo('✅ Smoking dataset processed and uploaded successfully!')


@app.command()
def tuberculosis(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload Tuberculosis (GSE19433) dataset to HuggingFace."""
    typer.echo('Processing Tuberculosis dataset...')
    process_tuberculosis(output_dir)
    typer.echo('✅ Tuberculosis dataset processed and uploaded successfully!')


@app.command()
def brca(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
) -> None:
    """Download and upload BRCA dataset to HuggingFace."""
    typer.echo('Processing BRCA dataset...')
    process_brca(output_dir)
    typer.echo('✅ BRCA dataset processed and uploaded successfully!')


@app.command()
def all(
    output_dir: str = typer.Option(
        'temp_data',
        '--output-dir',
        '-o',
        help='Output directory for temporary files',
    ),
    cleanup: bool = typer.Option(
        True,
        '--cleanup/--no-cleanup',
        help='Clean up temporary files after upload',
    ),
) -> None:
    """Download and upload all datasets to HuggingFace."""
    typer.echo('Processing all datasets...')

    try:
        # Process each dataset
        typer.echo('\n📊 Processing MotrPac...')
        process_motrpac(output_dir)

        typer.echo('\n📊 Processing AddNeuroMed...')
        process_addneuromed(output_dir)

        typer.echo('\n📊 Processing Parkinsons...')
        process_parkinsons(output_dir)

        typer.echo('\n📊 Processing Smoking...')
        process_smoking(output_dir)

        typer.echo('\n📊 Processing Tuberculosis...')
        process_tuberculosis(output_dir)

        typer.echo('\n📊 Processing BRCA...')
        process_brca(output_dir)

        typer.echo('\n✅ All datasets processed and uploaded successfully!')

        # Clean up temporary files
        if cleanup:
            typer.echo(f'\n🧹 Cleaning up temporary files in {output_dir}...')
            shutil.rmtree(output_dir, ignore_errors=True)
            typer.echo('✅ Cleanup completed!')

    except Exception as e:
        typer.echo(f'❌ Error processing datasets: {str(e)}')
        raise typer.Exit(1) from e


if __name__ == '__main__':
    app()
