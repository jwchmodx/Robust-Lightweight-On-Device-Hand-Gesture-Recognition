# Root Execution Verification

Verified from repository root:

```bash
cd /home/bae/project/gesture-integrated-repo
```

## Data copied

- `models/dynamic/robust/data/processed`
- `models/dynamic/robust/data/processed_baseline_static_conf05`
- `models/dynamic/robust/data/processed_tracking`
- `models/dynamic/robust/data/model_ready`
- `models/static/hagrid/npy_dataset`
- `models/static/hagrid/npy_experiments`

Approximate copied preprocessed data size: 4.4 GB.

## Passed checks

- Python syntax check for `demo`, `preprocessing`, and `models`
- Static frozen MLP contract evaluation
  - macro F1: `0.99114`
- Dynamic GRU F1 evaluation
  - `gru_control_best.pt`: macro F1 `0.816604`
  - `gru_armed12_24f_best.pt`: macro F1 `0.849661`
- Dynamic Armed-12 training script verify-only run
- Dynamic Armed-4 training script verify-only run
- Static HaGRID training script verify-only run
- Armed-12 gallery demo self-test
- Armed-12 webcam demo CLI help
- Extractor proxy metrics generation
- Extractor train and robustness sweep CLI help

## Excluded from execution

Dataset creation or raw-data preprocessing scripts were intentionally not run:

- `build_*dataset*.py`
- `preprocess_jester_mediapipe.py`
- extractor crop/manifest builders

Webcam live mode was not run because it requires camera and GUI access.
