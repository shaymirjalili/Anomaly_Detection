# Anomaly Detection Project (MVTec AD)

This project is a notebook-based anomaly detection workflow using:
- A **PaDiM baseline**
- A **Reverse Distillation StudentMethod** implementation
- Image-level and pixel-level evaluation
- Qualitative visual comparisons

Main notebook: `anomaly_detection_project_question.ipynb`

The current notebook runs a **30-epoch Reverse Distillation experiment** on all
15 MVTec categories. The previous VAE implementation and its completed 15-category
results are preserved in `anomaly_detection_vae.ipynb`.

Reverse Distillation uses a frozen ImageNet ResNet18 teacher, the authors'
one-class bottleneck, and a reverse ResNet18 decoder. Both PaDiM and this student
use ImageNet-normalized inputs. Training uses only `train/good`; cached float16
teacher features avoid repeated image decoding. This is a smaller experiment
than the paper's WideResNet50 / 200-epoch setup.

Set `EXPERIMENT_CATEGORIES = None` in the category-selection cell to run all
available categories. `StudentMethod` is defined in `reverse_distillation.py`;
the original MIT-licensed architecture files are in `third_party/rd4ad`.
Results, per-epoch losses, prediction arrays, checkpoints, and qualitative PNGs
are saved under `outputs/reverse_distillation/`.

Run the notebook top-to-bottom, or execute `python run_reverse_distillation.py`
in an environment with the dependencies below. Restore a trained model with
`StudentMethod.load('outputs/reverse_distillation/checkpoints/bottle.pt')`.
Prediction expects ImageNet-normalized 224 x 224 RGB images and returns raw
cosine-distance scores; choose an operating threshold using a separate normal
validation set before deploying it.

## Completed Results

The saved notebook includes the completed **15-category run**, with 30 training
epochs per category, seed 42, and 224 x 224 inputs. All 48 code cells completed
without cell errors. Training uses normal images only; the metrics below are
computed on the category test sets. Higher AUROC is better.

| Method | Mean image AUROC | Mean pixel AUROC |
|---|---:|---:|
| VAE (previous run) | 0.5647 | 0.6838 |
| PaDiM | 0.9458 | 0.9647 |
| Reverse Distillation | **0.9611** | **0.9693** |

These are category means using the notebook's evaluator. Pixel AUROC follows
its per-anomalous-image averaging, rather than pooling every test pixel.
This is a local ResNet18 experiment, not a reproduction of the paper's benchmark.

- [Executed notebook](anomaly_detection_project_question.ipynb): sections 8–10
  contain metrics and figures; the last cell compares the VAE and Reverse Distillation.
- [Reverse Distillation metrics](outputs/reverse_distillation/rd_metrics.csv)
- [PaDiM comparison](outputs/reverse_distillation/padim_vs_rd.csv)
- [Previous VAE comparison](outputs/reverse_distillation/vae_vs_rd.csv)
- [Saved heatmaps](outputs/reverse_distillation/qualitative/)

The repository includes executed notebooks, CSV metrics, per-epoch training
histories, prediction arrays, and qualitative figures. Dataset images, model
checkpoints, caches, and temporary execution logs are excluded from Git.
Rerunning the notebook creates the model checkpoints locally.
The student's `runtime_sec` measures prediction only, while PaDiM's includes its
full category run; use the separate training JSON timings when comparing cost.

## Project Structure

Expected workspace layout:

```text
Anomaly_Detection/
|-- anomaly_detection_project_question.ipynb
|-- mvtec_anomaly_detection/
|   |-- bottle/
|   |-- cable/
|   `-- ... (MVTec categories)
`-- outputs/
    `-- reverse_distillation/
        `-- qualitative/    # Reverse Distillation figures; previous VAE figures remain in outputs/qualitative
```

## Requirements

Install Python dependencies (example):

```powershell
pip install numpy pandas scipy matplotlib pillow scikit-learn torch torchvision jupyter nbclient nbformat ipykernel
```

## Dataset Setup

1. Download **MVTec AD** dataset archive (`mvtec_anomaly_detection.tar.xz`).
2. Extract it so you get:

```text
./mvtec_anomaly_detection/
```

The notebook uses:
- `DATASET_ROOT = Path('./mvtec_anomaly_detection')`

## How To Run

1. Start Jupyter:

```powershell
jupyter notebook
```

2. Open `anomaly_detection_project_question.ipynb`.
3. Run cells top-to-bottom.

Notebook flow:
- `0) Setup`
- `1) Download the dataset (MVTec AD)`
- `2) Dataset utilities`
- `3) Dataset + DataLoaders`
- `4) Dataset sanity checks`
- `5) PaDiM baseline implementation`
- `6) Student implementation`
- `7) Metrics + Reporting utilities`
- `8) Run models`
- `9) PaDiM vs Student comparison`
- `10) Competitive leaderboard metrics`

## Student Output Contract

`StudentMethod.predict(test_loader)` output must align with test loader order and include:
- `paths`
- `scores`
- `score_maps`
- `runtime_sec`

Per-category outputs are stored as:

```python
student_out_by_cat[category] = student.predict(test_loader)
```

## Saved Output Images

Qualitative figures are automatically saved to:

```text
./outputs/reverse_distillation/qualitative/
```

Current plotting helpers support saving:
- Per-model qualitative figures (PaDiM / Student)
- Integrated `PaDiM vs Student` overlays

If needed, you can override save location with the plotting function arguments:
- `save_dir`
- `save_dpi`

## Notes

- Random seed is set in the notebook (`SEED = 42`) for reproducibility.
- DataLoader uses `shuffle=False` for evaluation ordering consistency.
- If you change output dictionaries, keep path/score/map alignment intact.
