#!/bin/bash -l

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

    # Install Miniconda silently
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"

    # Remove installer
    rm /tmp/miniconda.sh

    # Initialize conda for bash shell
    eval "$("${HOME}"/miniconda/bin/conda shell.bash hook)"

    # Add conda to PATH permanently
    echo "export PATH=${HOME}/miniconda/bin:$PATH" >> ~/.bashrc

    # Source bashrc to apply changes
    # shellcheck disable=SC1090
    source ~/.bashrc

    echo "Conda installation complete"
fi

if ! conda env list | grep -q "ogbench"; then
    conda create -n ogbench python=3.12 -y
fi

conda activate ogbench

pip install --upgrade pip
pip install -e '.[all]'

#pre-commit install
