# DUBNF Air Quality Prediction

This folder contains the implementation used for spatiotemporal air-quality prediction with a heteroscedastic Bayesian neural-field ensemble.

## Files

- `models.py`: neural-field layers, feature encoders, likelihood construction, and uncertainty heads.
- `inference.py`: MAP ensemble training and batched prediction utilities.
- `spatiotemporal.py`: data preprocessing, estimator classes, prediction export, calibration metrics, and plotting helpers.
- `train_air_quality.py`: command-line training and evaluation entry point.

## Setup

```bash
pip install -r requirements.txt
```

## Example

```bash
python train_air_quality.py --data-dir ../datesets/original_data --output-dir ../datesets/results/dubnf
```

The default script expects train/test CSV files with `datetime`, `latitude`, `longitude`, and `pm10` columns. Use `--train-csv` and `--test-csv` for a single explicit split, or `--split-ids` with filename templates for batch evaluation.

## License Notes

Files that retain Apache 2.0 headers must keep those notices when redistributed.
