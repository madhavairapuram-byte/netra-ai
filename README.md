# EyeCheck AI — Conjunctivitis Screening Prototype

A mobile-first educational prototype for your AI in Healthcare project.

## Current build

- Mobile-first camera/image capture UI
- 9-slot workflow: 3 days × morning/afternoon/evening
- Image quality check
- Transparent experimental redness heuristic so the demo runs without a trained medical model
- Per-image screening probability
- 9-image aggregation endpoint
- Training script for a real EfficientNet-B0 classifier
- No image persistence by default
- Prominent non-diagnostic safety language

## Important medical limitation

The included redness heuristic is **not clinically validated** and must not be presented as a conjunctivitis diagnostic model. It exists only so the application flow can be demonstrated before you have a properly labelled clinical dataset.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

For phone testing on the same Wi-Fi, use the computer's LAN IP, for example:

```text
http://192.168.1.20:8000
```

Camera access may require a secure context depending on the browser. The local development browser can still use the image picker path if direct camera capture is unavailable.

## Train the real model

Prepare the patient-level split described in `data/README.md`, then run:

```bash
python training/train.py --data data --epochs 8
```

The output model is written to:

```text
backend/conjunctivitis_model.pt
```

The backend already has a model hook, but the inference adapter still needs to be switched from the demo heuristic to the trained PyTorch model after you establish proper preprocessing and validation.

## Next research milestone

Compare:

1. single-image screening
2. 9-image multi-timepoint screening

Report sensitivity, specificity, F1, AUROC, and calibration on a patient-held-out test set.
