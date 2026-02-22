# Anomaly Detection Project (MVTec AD)

This project is a notebook-based anomaly detection workflow using:
- A **PaDiM baseline**
- A **StudentMethod** implementation
- Image-level and pixel-level evaluation
- Qualitative visual comparisons

Main notebook: `anomaly_detection_project_question.ipynb`

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
    `-- qualitative/    # generated qualitative figures
```

## Requirements

Install Python dependencies (example):

```powershell
pip install numpy pandas matplotlib pillow scikit-learn torch torchvision jupyter
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
./outputs/qualitative/
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
