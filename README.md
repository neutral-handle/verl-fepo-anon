# FEPO Minimal Reproduction (Anonymous)

This repository contains only the minimum assets required to reproduce FEPO:

- `patches/fepo_core.patch`: FEPO core patch against upstream `verl`
- `patches/base_commit.txt`: base commit reference from the local `verl` snapshot
- `scripts/run_fepo_minimal.sh`: anonymized minimal training launcher (no personal paths)

## 1) Prepare upstream code

```bash
git clone https://github.com/volcengine/verl.git
cd verl
```

Optional: if you want closer environment alignment, check `base_commit.txt` and switch to a nearby upstream commit.

## 2) Apply FEPO patch

Assume your current directory is upstream `verl`, and this repository is located at `../verl-fepo-anon`:

```bash
git apply ../verl-fepo-anon/patches/fepo_core.patch
```

## 3) Launch training (example)

```bash
bash ../verl-fepo-anon/scripts/run_fepo_minimal.sh
```

All data paths, model paths, and output directories are injected via environment variables, with no hardcoded personal information.

## 4) Key environment variables

- `MODEL_PATH`: base model path
- `TRAIN_FILES`: training parquet file
- `AIME24_VAL`, `AIME25_VAL`: validation parquet files
- `OUTPUT_DIR`: experiment output directory
- `WANDB_API_KEY`: optional, injected externally

## Notes

- This is a patch-and-script minimal reproduction repository and does not include full `verl` source code.
- You can publish this repository under your anonymous GitHub account and state that it is based on upstream `verl`.
