"""Training entry point for air-quality DUBNF experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np
import pandas as pd

try:
    from .spatiotemporal import (
        BayesianNeuralFieldMAP,
        plot_coverage_diagram,
        plot_reliability_diagram,
        plot_uncertainty_error_scatter,
        split_metrics_for_export,
    )
except ImportError:
    from spatiotemporal import (
        BayesianNeuralFieldMAP,
        plot_coverage_diagram,
        plot_reliability_diagram,
        plot_uncertainty_error_scatter,
        split_metrics_for_export,
    )


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "datesets"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "datesets" / "results" / "london260407"
DEFAULT_TRAIN_TEMPLATE = "air_quality.{split_id}.train.csv"
DEFAULT_TEST_TEMPLATE = "air_quality.{split_id}.test.csv"
DEFAULT_SPLIT_IDS = (5, 6, 7, 8, 9)


DEFAULT_QUANTILES = (0.025, 0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95, 0.975)
DEFAULT_COVERAGE_LEVELS = (0.5, 0.7, 0.9, 0.95)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _save_metrics_json(metrics: dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fp:
        json.dump(_json_ready(metrics), fp, indent=2, ensure_ascii=False)


def _save_metrics_csv(metrics: dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_json_ready(metrics)]).to_csv(save_path, index=False)


def _build_model(args: argparse.Namespace) -> BayesianNeuralFieldMAP:
    return BayesianNeuralFieldMAP(
        width=512,
        depth=3,
        freq="h",
        interactions=[(0, 1), (0, 2), (1, 2)],
        seasonality_periods=["D", "W", "M"],
        num_seasonal_harmonics=[6, 4, 3],
        feature_cols=["datetime", "latitude", "longitude"],
        target_col="pm10",
        observation_model="NORMAL",
        timetype="index",
        standardize=["latitude", "longitude"],
        min_log_variance=args.min_log_variance,
        max_log_variance=args.max_log_variance,
        variance_output_init_scale=args.variance_output_init_scale,
        detach_variance_head=args.detach_variance_head,
        variance_regularization_weight=args.variance_regularization_weight,
        variance_reference_scale_factor=args.variance_reference_scale_factor,
    )


def _sanitize_output_label(label: Any) -> str:
    label = str(label).strip()
    if not label:
        raise ValueError("split label must be non-empty.")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label)


def _infer_split_label(
    train_csv: Path,
    test_csv: Path,
    explicit_label: str | None = None,
) -> str:
    if explicit_label is not None:
        return _sanitize_output_label(explicit_label)

    pattern = re.compile(r"air_quality\.(\d+)\.(?:train|test)$")
    for csv_path in (train_csv, test_csv):
        match = pattern.search(csv_path.stem)
        if match:
            return match.group(1)
    return _sanitize_output_label(train_csv.stem)


def _build_output_prefix(split_label: str) -> str:
    split_label = _sanitize_output_label(split_label)
    if split_label.isdigit():
        return f"air_quality.{split_label}"
    return split_label


def _build_output_paths(output_dir: Path, split_label: str) -> dict[str, Path]:
    prefix = _build_output_prefix(split_label)
    return {
        "prediction_csv": output_dir / f"{prefix}.predictions.csv",
        "metrics_json": output_dir / f"{prefix}.metrics.json",
        "metrics_csv": output_dir / f"{prefix}.metrics.csv",
        "calibration_json": output_dir / f"{prefix}.calibration_metrics.json",
        "calibration_csv": output_dir / f"{prefix}.calibration_metrics.csv",
        "reliability_png": output_dir / f"{prefix}.reliability_diagram.png",
        "coverage_png": output_dir / f"{prefix}.coverage_plot.png",
        "uncertainty_scatter_png": output_dir
        / f"{prefix}.uncertainty_error_scatter.png",
    }


def _resolve_split_paths(
    data_dir: Path,
    train_template: str,
    test_template: str,
    split_id: int,
) -> tuple[Path, Path]:
    try:
        train_csv = data_dir / train_template.format(split_id=split_id)
        test_csv = data_dir / test_template.format(split_id=split_id)
    except KeyError as exc:
        raise ValueError(
            "train/test template must include the `{split_id}` placeholder."
        ) from exc
    return train_csv, test_csv


def _build_run_jobs(
    args: argparse.Namespace,
) -> list[tuple[str, Path, Path]]:
    if args.train_csv is not None and args.test_csv is not None:
        train_csv = Path(args.train_csv)
        test_csv = Path(args.test_csv)
        split_label = _infer_split_label(train_csv, test_csv, args.split_label)
        return [(split_label, train_csv, test_csv)]

    jobs = []
    data_dir = Path(args.data_dir)
    for split_id in args.split_ids:
        train_csv, test_csv = _resolve_split_paths(
            data_dir=data_dir,
            train_template=args.train_template,
            test_template=args.test_template,
            split_id=split_id,
        )
        jobs.append((str(split_id), train_csv, test_csv))
    return jobs


def _print_split_summary(
    split_label: str,
    summary_metrics: dict[str, Any],
    calibration_metrics: dict[str, Any],
) -> None:
    def _format_metric(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6f}"

    mae = summary_metrics.get("mae")
    rmse = summary_metrics.get("rmse")
    r2 = summary_metrics.get("r2")
    nll = calibration_metrics.get("nll")
    ece = calibration_metrics.get("ece")
    print(
        f"Split {split_label} metrics: "
        f"MAE={_format_metric(mae)}, "
        f"RMSE={_format_metric(rmse)}, "
        f"R2={_format_metric(r2)}, "
        f"NLL={_format_metric(nll)}, "
        f"ECE={_format_metric(ece)}"
    )
    interval_parts = []
    for coverage_level in DEFAULT_COVERAGE_LEVELS:
        label = int(round(float(coverage_level) * 100.0))
        picp = calibration_metrics.get(f"picp_{label}")
        mpiw = calibration_metrics.get(f"mpiw_{label}")
        if picp is None and mpiw is None:
            continue
        interval_parts.append(
            f"{label}%: "
            f"PICP={_format_metric(picp)}, "
            f"MPIW={_format_metric(mpiw)}"
        )
    if interval_parts:
        print("Interval metrics: " + "; ".join(interval_parts))


def run_one_split(
    split_label: str,
    train_csv: Path,
    test_csv: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"===== Running split {split_label} =====")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")

    result_row: dict[str, Any] = {
        "split_id": split_label,
        "status": "failed",
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
    }

    missing_paths = [path for path in (train_csv, test_csv) if not path.exists()]
    if missing_paths:
        error_message = "Missing input file(s): " + ", ".join(
            str(path) for path in missing_paths
        )
        print(f"[ERROR] {error_message}")
        result_row["status"] = "missing_input"
        result_row["error_message"] = error_message
        return result_row

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _build_output_paths(output_dir, split_label)

    try:
        df_train = pd.read_csv(train_csv, index_col=0, parse_dates=["datetime"])
        df_test = pd.read_csv(test_csv, index_col=0, parse_dates=["datetime"])

        model = _build_model(args)
        train_start = time.time()
        model = model.fit(
            df_train,
            seed=jax.random.PRNGKey(args.seed),
            ensemble_size=args.ensemble_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        )
        training_time_seconds = float(time.time() - train_start)

        metrics, prediction_df = model.evaluate_dataframe(
            df_test,
            quantiles=DEFAULT_QUANTILES,
            prediction_col=args.prediction_col,
            coverage_levels=DEFAULT_COVERAGE_LEVELS,
        )
        summary_metrics, calibration_metrics = split_metrics_for_export(metrics)

        summary_metrics.update(
            {
                "split_id": split_label,
                "training_time_seconds": training_time_seconds,
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "output_dir": str(output_dir),
                "ensemble_size": args.ensemble_size,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "uncertainty_model": "heteroscedastic_gaussian_ensemble",
                "noise_model": "heteroscedastic_gaussian",
                "aleatoric_mode": "input_dependent",
                "min_log_variance": args.min_log_variance,
                "max_log_variance": args.max_log_variance,
                "variance_output_init_scale": args.variance_output_init_scale,
                "variance_regularization_weight": args.variance_regularization_weight,
                "variance_reference_scale_factor": args.variance_reference_scale_factor,
                "detach_variance_head": args.detach_variance_head,
                "heteroscedastic_stability_mode": (
                    "bounded_log_variance_detached_head"
                    if args.detach_variance_head
                    else "bounded_log_variance_shared_head"
                ),
            }
        )
        calibration_metrics.update(
            {
                "split_id": split_label,
                "prediction_csv": str(output_paths["prediction_csv"]),
                "reliability_diagram_png": str(output_paths["reliability_png"]),
                "coverage_plot_png": str(output_paths["coverage_png"]),
                "uncertainty_error_scatter_png": str(
                    output_paths["uncertainty_scatter_png"]
                ),
                "noise_model": "heteroscedastic_gaussian",
                "aleatoric_mode": "input_dependent",
                "min_log_variance": args.min_log_variance,
                "max_log_variance": args.max_log_variance,
                "variance_output_init_scale": args.variance_output_init_scale,
                "variance_regularization_weight": args.variance_regularization_weight,
                "variance_reference_scale_factor": args.variance_reference_scale_factor,
                "detach_variance_head": args.detach_variance_head,
            }
        )

        prediction_df.to_csv(output_paths["prediction_csv"], index=False)
        _save_metrics_json(summary_metrics, output_paths["metrics_json"])
        _save_metrics_csv(summary_metrics, output_paths["metrics_csv"])
        _save_metrics_json(calibration_metrics, output_paths["calibration_json"])
        _save_metrics_csv(calibration_metrics, output_paths["calibration_csv"])

        target_col = model.evaluation_target_col
        plot_reliability_diagram(
            prediction_df,
            prediction_col=args.prediction_col,
            target_col=target_col,
            save_path=output_paths["reliability_png"],
        )
        plot_coverage_diagram(
            prediction_df,
            prediction_col=args.prediction_col,
            target_col=target_col,
            save_path=output_paths["coverage_png"],
            coverage_levels=DEFAULT_COVERAGE_LEVELS,
        )
        plot_uncertainty_error_scatter(
            prediction_df,
            prediction_col=args.prediction_col,
            target_col=target_col,
            save_path=output_paths["uncertainty_scatter_png"],
            error_kind="absolute_error",
        )

        print(f"Training time: {training_time_seconds:.2f} seconds")
        _print_split_summary(split_label, summary_metrics, calibration_metrics)
        print(f'Predictions saved to: {output_paths["prediction_csv"]}')
        print(
            "Metrics saved to: "
            f'{output_paths["metrics_json"]} and {output_paths["metrics_csv"]}'
        )
        print(
            "Calibration metrics saved to: "
            f'{output_paths["calibration_json"]} and {output_paths["calibration_csv"]}'
        )
        print(f'Reliability diagram saved to: {output_paths["reliability_png"]}')
        print(f'Coverage plot saved to: {output_paths["coverage_png"]}')
        print(
            "Uncertainty-error scatter saved to: "
            f'{output_paths["uncertainty_scatter_png"]}'
        )

        result_row.update(_json_ready(summary_metrics))
        result_row.update(_json_ready(calibration_metrics))
        result_row["status"] = "success"
        return result_row
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] Split {split_label} failed: {error_message}")
        traceback.print_exc()
        result_row["error_message"] = error_message
        return result_row


def _save_all_metrics_summary(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    if not results:
        return None

    summary_csv = output_dir / "all_metrics_summary.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False)
    return summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the DUBNF heteroscedastic neural-field model and "
            "export uncertainty / calibration artifacts."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-template", type=str, default=DEFAULT_TRAIN_TEMPLATE)
    parser.add_argument("--test-template", type=str, default=DEFAULT_TEST_TEMPLATE)
    parser.add_argument(
        "--split-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_SPLIT_IDS),
        help="Split ids to run in batch mode. Default: 5 6 7 8 9.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Optional explicit training CSV. Must be paired with --test-csv.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Optional explicit test CSV. Must be paired with --train-csv.",
    )
    parser.add_argument(
        "--split-label",
        type=str,
        default=None,
        help="Optional output label used in single-run mode.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-col", type=str, default="predict_pm10")
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-log-variance", type=float, default=-8.0)
    parser.add_argument("--max-log-variance", type=float, default=6.0)
    parser.add_argument("--variance-output-init-scale", type=float, default=0.05)
    parser.add_argument("--variance-regularization-weight", type=float, default=1e-4)
    parser.add_argument("--variance-reference-scale-factor", type=float, default=0.5)
    parser.add_argument(
        "--detach-variance-head",
        dest="detach_variance_head",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-detach-variance-head",
        dest="detach_variance_head",
        action="store_false",
    )
    args = parser.parse_args()
    if (args.train_csv is None) != (args.test_csv is None):
        parser.error("--train-csv and --test-csv must be provided together.")
    if args.train_csv is not None and args.split_label is not None:
        args.split_label = _sanitize_output_label(args.split_label)
    if not args.split_ids:
        parser.error("--split-ids cannot be empty.")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_jobs = _build_run_jobs(args)
    results = []
    for split_label, train_csv, test_csv in run_jobs:
        results.append(
            run_one_split(
                split_label=split_label,
                train_csv=train_csv,
                test_csv=test_csv,
                args=args,
            )
        )

    summary_csv = _save_all_metrics_summary(results, output_dir)
    if summary_csv is not None:
        success_count = sum(result.get("status") == "success" for result in results)
        failure_count = len(results) - success_count
        print(
            f"Completed {len(results)} split(s): {success_count} succeeded, {failure_count} failed."
        )
        print(f"All metrics summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
