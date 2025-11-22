#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8xH100 node at $3/GPU/hour.
# The script auto-detects available GPUs and scales accordingly.

# 1) Example launch (simplest, auto-detects GPUs):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 4) Example launch in Docker (auto-detects pre-installed PyTorch, avoids CUDA conflicts):
# docker run --gpus all -it --rm --ipc=host -v $HOME/.cache/nanochat:/root/.cache/nanochat -v ${PWD}:/workspace -w /workspace nvcr.io/nvidia/pytorch:25.09-py3 bash speedrun.sh
# 5) Force specific number of GPUs (override auto-detection):
# NPROC_PER_NODE=4 bash speedrun.sh
#
# Note: The script detects if PyTorch with CUDA is already installed (e.g., in a container)
# and skips venv creation to use the pre-installed version, avoiding CUDA reinstallation.

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Suppress PyTorch's pynvml deprecation warning (PyTorch's own dependency issue)
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda"

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env file"
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

# Detect if we're in a container with PyTorch already installed (e.g., NVIDIA PyTorch container)
# If so, skip venv creation and use the system Python to avoid CUDA conflicts
PYTORCH_PREINSTALLED=false
# Clean up any broken venv from previous runs if we detect PyTorch is pre-installed
if python -c "import torch" 2>/dev/null && python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [ -d ".venv" ]; then
        echo "Detected pre-installed PyTorch and existing .venv - removing .venv to avoid conflicts..."
        rm -rf .venv
    fi
fi
PYTORCH_PREINSTALLED=false
if python -c "import torch" 2>/dev/null && python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Detected PyTorch with CUDA already installed (likely running in container)"
    echo "Skipping venv creation to use pre-installed PyTorch and avoid CUDA conflicts"
    PYTORCH_PREINSTALLED=true
fi

if [ "$PYTORCH_PREINSTALLED" = false ]; then
    # install uv (if not already installed)
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to PATH for this session
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    # create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv
    # install the repo dependencies (this will install PyTorch with CUDA)
    uv sync --extra gpu
    # activate venv so that `python` uses the project's venv instead of system python
    source .venv/bin/activate
else
    # Install only non-PyTorch dependencies using system Python
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    # Install dependencies without the gpu extra (which includes PyTorch)
    # Use pip to install all required packages from pyproject.toml except torch
    # Also install nvidia-ml-py to replace deprecated pynvml
    pip install -q datasets fastapi files-to-prompt psutil regex setuptools tiktoken tokenizers uvicorn wandb maturin nvidia-ml-py
    echo "Installed dependencies using system Python with pre-installed PyTorch"
fi

# -----------------------------------------------------------------------------
# GPU Prerequisites Check

echo "Checking GPU prerequisites..."

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU drivers may not be installed properly."
    echo "If running in Docker, make sure to use --gpus all flag."
    exit 1
fi

# Check if GPUs are available
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Please check your GPU setup."
    exit 1
fi

echo "Found $GPU_COUNT GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Verify PyTorch can see GPUs
echo "Verifying PyTorch GPU access..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'; print(f'PyTorch can see {torch.cuda.device_count()} GPU(s)'); print(f'CUDA Version: {torch.version.cuda}')" || {
    echo "ERROR: PyTorch cannot access CUDA GPUs"
    exit 1
}

# Auto-detect and set number of GPUs to use
# Override with NPROC_PER_NODE environment variable if set
if [ -z "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$GPU_COUNT
    echo "Auto-detected $NPROC_PER_NODE GPU(s) for training"
else
    echo "Using $NPROC_PER_NODE GPU(s) (set via NPROC_PER_NODE environment variable)"
fi

# Verify we don't try to use more GPUs than available
if [ "$NPROC_PER_NODE" -gt "$GPU_COUNT" ]; then
    echo "WARNING: Requested $NPROC_PER_NODE GPUs but only $GPU_COUNT available. Using $GPU_COUNT instead."
    NPROC_PER_NODE=$GPU_COUNT
fi

echo "GPU prerequisites check passed ✓"
echo ""

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# The WANDB_API_KEY is loaded from .env file automatically.
# Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=speedrun bash speedrun.sh`
# If WANDB_API_KEY is set and WANDB_RUN is not "dummy", wandb will be used for tracking.
if [ -z "$WANDB_RUN" ]; then
    if [ -n "$WANDB_API_KEY" ]; then
        # If API key is set but no run name specified, use a default run name
        WANDB_RUN="speedrun-$(date +%Y%m%d-%H%M%S)"
        echo "Using wandb for tracking with run name: $WANDB_RUN"
    else
        # by default use "dummy" : it's handled as a special case, skips logging to wandb
        WANDB_RUN=dummy
        echo "No WANDB_API_KEY found, skipping wandb logging"
    fi
else
    if [ -n "$WANDB_API_KEY" ] && [ "$WANDB_RUN" != "dummy" ]; then
        echo "Using wandb for tracking with run name: $WANDB_RUN"
    fi
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (if not already installed)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    # Ensure cargo is in PATH even if already installed
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Build the rustbpe Tokenizer
if [ "$PYTORCH_PREINSTALLED" = true ]; then
    # In container with pre-installed PyTorch, build wheel and install with pip
    echo "Building tokenizer with maturin (using system Python)..."
    cd rustbpe
    # Clean old wheels to avoid version conflicts
    rm -rf target/wheels
    maturin build --release
    pip install --force-reinstall target/wheels/*.whl
    cd ..
else
    # On native system with venv, use uv run
    uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use (auto-detected above in GPU prerequisites check)
# You can override by setting NPROC_PER_NODE environment variable before running this script

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

# pretrain the d20 model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
# Clear GPU memory between stages
python -c "import torch; torch.cuda.empty_cache()"
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
