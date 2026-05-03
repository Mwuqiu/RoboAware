# Project Monorepo

This repository contains two related subprojects:

- `cosmos-predict2.5`: main video/world-model training and inference code.
- `Pointcept`: point-cloud encoder used by the main project.

## What is intentionally excluded

To keep the repository pushable and clean, the following are ignored:

- datasets and generated data
- virtual environments and local caches
- experiment outputs (`exp/`, `wandb/`, etc.)
- large local archives (`*.zip`)

## Environment setup

### cosmos-predict2.5 (uv / venv)

```bash
cd cosmos-predict2.5
source .venv/bin/activate
```

A lock-style dependency snapshot is provided at:

- `cosmos-predict2.5/requirements.lock.txt`

Install from snapshot in a fresh environment:

```bash
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

### Pointcept

Use the provided conda environment file:

```bash
cd Pointcept
conda env create -f environment.yml
conda activate pointcept
```
