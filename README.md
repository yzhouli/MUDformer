# MUDformer

MUDformer combines PoE-MVAE with a prior expert, two-stage sliding-window/window attention, user detection, and a behavior audit head. This repository provides preprocessing, data loading, training, checkpoint saving, and inference code, together with four trained checkpoints.

## 1. Installation

Use Python 3.10 or newer. Install a PyTorch build compatible with your GPU, then run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

All paths below are relative to the repository root. GPU examples use `--device cuda`; CPU execution is available with `--device cpu`.

## 2. Download and organize the data

### Raw datasets

The raw datasets are available from the following Kaggle pages:

| Dataset | Download page | Raw archive size | Extracted directory |
|---|---|---:|---|
| Weibo-25+ | [yangzhou32/weibo-25](https://www.kaggle.com/datasets/yangzhou32/weibo-25) | approximately 8.99 GB | `data/raw/Weibo-25-plus/` |
| MORSE | [yangzhou32/morse1](https://www.kaggle.com/datasets/yangzhou32/morse1) | approximately 121.02 GB | `data/raw/MORSE/` |

Download the original dataset archives from these pages. If Kaggle wraps the original archive in an additional download ZIP, extract the outer ZIP first. Place the original archives under `data/downloads/`, then extract them:

```bash
mkdir -p data/raw
python -m zipfile -e data/downloads/Weibo-25-plus_public_raw.zip data/raw
python -m zipfile -e data/downloads/MORSE_public_raw.zip data/raw
```

The required layout is:

```text
data/
├── downloads/
│   ├── Weibo-25-plus_public_raw.zip
│   └── MORSE_public_raw.zip
├── raw/
│   ├── Weibo-25-plus/
│   │   ├── users.jsonl
│   │   ├── relations.jsonl
│   │   ├── histories/
│   │   └── images/
│   └── MORSE/
│       ├── users.jsonl
│       ├── relations.jsonl
│       ├── histories/
│       └── images/
└── pretrained/
    ├── bert-base-chinese/
    └── vit-base-patch16-224/
```

Each line in `users.jsonl` describes an anonymous user, their label, profile, history path, and behavior count. Histories contain chronological behavior text and image paths. `relations.jsonl` stores user relations. Label 0 denotes normal users and label 1 denotes malicious users.

### Pretrained feature encoders

Download the following models, including their configuration and tokenizer/image-processor files:

| Modality | Pretrained model | Local directory |
|---|---|---|
| Text | [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) | `data/pretrained/bert-base-chinese/` |
| Image | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) | `data/pretrained/vit-base-patch16-224/` |

Download through the model pages or run this explicit download step using [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/download):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

for repo, folder in [
    ("google-bert/bert-base-chinese", "data/pretrained/bert-base-chinese"),
    ("google/vit-base-patch16-224", "data/pretrained/vit-base-patch16-224"),
]:
    snapshot_download(
        repo_id=repo,
        local_dir=folder,
        allow_patterns=["*.json", "*.txt", "*.safetensors"],
    )
PY
```

The text directory must contain `config.json`, `model.safetensors`, and tokenizer files, including `vocab.txt`. The image directory must contain `config.json`, `model.safetensors`, and `preprocessor_config.json`. Preprocessing reads local files and does not automatically download models. Both encoders produce 768-dimensional CLS features.

### Included MUDformer checkpoints

The four model weights are already included in the repository:

```text
checkpoints/
├── Weibo-25-plus/
│   ├── classification.pt
│   ├── audit.pt
│   └── calibration.json
└── MORSE/
    ├── classification.pt
    ├── audit.pt
    └── calibration.json
```

Pretrained classification and behavior-audit checkpoints are provided for both datasets.

## 3. Preprocess and load a dataset

Select a dataset once in your shell. Use `MORSE` or `Weibo-25-plus`:

```bash
MUD_DATASET=MORSE
MUD_RAW="data/raw/$MUD_DATASET"
MUD_WORK="data/processed/$MUD_DATASET"
MUD_RUN="outputs/$MUD_DATASET"
```

Generate a user-stratified **8:1:1** train/validation/test split, extract behavior features, and prepare profile/graph features:

```bash
python scripts/build_split.py \
  --users "$MUD_RAW/users.jsonl" \
  --output-dir "$MUD_WORK/splits" --seed 42

python scripts/preprocess.py \
  --raw-root "$MUD_RAW" --output-dir "$MUD_WORK/features" \
  --text-model data/pretrained/bert-base-chinese \
  --image-model data/pretrained/vit-base-patch16-224 \
  --device cuda --batch-size 32

python scripts/prepare_auxiliary.py \
  --raw-root "$MUD_RAW" --splits "$MUD_WORK/splits" \
  --output-dir "$MUD_WORK/aux" --seed 42
```

The resulting structure is:

```text
data/processed/<dataset>/
├── splits/
│   ├── train.json
│   ├── validation.json
│   └── test.json
├── features/
│   ├── preprocessing.json
│   ├── u00000.npz
│   └── ...
└── aux/
    └── auxiliary.npz
```

Each user NPZ stores 768-dimensional text and image features with availability masks. `mudformer/data.py` loads these features together with user profiles and graph features, then crops or pads histories to the specified length.

## 4. Train and save models

### User classification

```bash
python train.py \
  --config configs/full.json --feature-root "$MUD_WORK/features" \
  --train-manifest "$MUD_WORK/splits/train.json" \
  --validation-manifest "$MUD_WORK/splits/validation.json" \
  --test-manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --output-dir "$MUD_RUN/classification" --seed 42 \
  --max-length 2048 --batch-size 2 --epochs 500 --patience 10 \
  --learning-rate 0.0001 --weight-decay 0.0001
```

Validation accuracy determines early stopping. The selected model is saved to `$MUD_RUN/classification/final.pt`; `latest.pt` supports automatic training resume.

### Behavior audit head

Initialize from the classification checkpoint and train the audit head with the backbone frozen:

```bash
python scripts/train_temporal_mil_mudformer.py \
  --config configs/full.json \
  --initialize-checkpoint "$MUD_RUN/classification/final.pt" --freeze-base \
  --feature-root "$MUD_WORK/features" \
  --train-manifest "$MUD_WORK/splits/train.json" \
  --validation-manifest "$MUD_WORK/splits/validation.json" \
  --test-manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --output-dir "$MUD_RUN/audit" --seed 42 \
  --max-length 16384 --batch-size 1 --epochs 50 --patience 10 \
  --learning-rate 0.0001 --weight-decay 0.0001 \
  --mvae-weight 0 --clean-distillation-weight 3 --attribution-block-weight 1
```

The selected audit model is saved to `$MUD_RUN/audit/final.pt`.

## 5. Inference and evaluation metrics

### User classification

```bash
python evaluate.py \
  --checkpoint "$MUD_RUN/classification/final.pt" \
  --feature-root "$MUD_WORK/features" \
  --manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --max-length 2048 --batch-size 1 --device cuda \
  --metrics-output "$MUD_RUN/classification_metrics.json" \
  --predictions-output "$MUD_RUN/classification_predictions.json"
```

Accuracy and macro precision, recall, and F1 are saved to `$MUD_RUN/classification_metrics.json`. To use the included weights, set `--checkpoint "checkpoints/$MUD_DATASET/classification.pt"`.

### Behavior localization

```bash
python audit.py \
  --checkpoint "$MUD_RUN/audit/final.pt" \
  --feature-root "$MUD_WORK/features" \
  --manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --max-length 16384 --device cuda \
  --output "$MUD_RUN/behavior_blocks.jsonl"
```

The output records five chronological block scores and the highest-risk block. For the included weights, use `--checkpoint "checkpoints/$MUD_DATASET/audit.pt"`.

### IDR and TGLS@1

Calibrate on training users, generate clean predictions, then evaluate controlled block replacement:

```bash
python scripts/calibrate_mudformer_temporal_threshold.py \
  --checkpoint "$MUD_RUN/audit/final.pt" \
  --feature-root "$MUD_WORK/features" \
  --train-manifest "$MUD_WORK/splits/train.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --seed 42 --output-dir "$MUD_RUN/calibration" --device cuda

python evaluate.py \
  --checkpoint "$MUD_RUN/audit/final.pt" \
  --feature-root "$MUD_WORK/features" \
  --manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --logit-key audit_logits --max-length 16384 --batch-size 1 --device cuda \
  --predictions-output "$MUD_RUN/clean_audit_predictions.json"

python scripts/evaluate_mudformer_temporal_localization.py \
  --checkpoint "$MUD_RUN/audit/final.pt" \
  --clean-predictions "$MUD_RUN/clean_audit_predictions.json" \
  --feature-root "$MUD_WORK/features" \
  --train-manifest "$MUD_WORK/splits/train.json" \
  --test-manifest "$MUD_WORK/splits/test.json" \
  --aux-features "$MUD_WORK/aux/auxiliary.npz" \
  --decision-calibration "$MUD_RUN/calibration/calibration.json" \
  --logit-key audit_logits --auxiliary-temporal-training \
  --seed 42 --output-dir "$MUD_RUN/localization" --batch-size 1 --device cuda
```

IDR and TGLS@1 are saved to `$MUD_RUN/localization/summary.json`. All metrics are reported on a 0–1 scale.
