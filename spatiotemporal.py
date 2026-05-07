"""API for Bayesian Neural Field estimators."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow_probability.substrates.jax as tfp
from flax import linen as nn
from sklearn.neighbors import NearestNeighbors

try:
    from . import inference
except ImportError:
    import inference


tfd = tfp.distributions
EPS = 1e-8
MIN_STD = 1e-6
DEFAULT_COVERAGE_LEVELS = (0.5, 0.7, 0.9, 0.95)
DEFAULT_CALIBRATION_BINS = 10


def seasonality_to_float(seasonality: str, freq: str) -> float:
    """Convert a valid pandas frequency string to a float, relative `freq`."""
    four_years = pd.date_range("2020-01-01", periods=5, freq="YS")
    y = four_years.to_period(seasonality)
    num_seasonality = (y[-1] - y[0]).n
    x = pd.date_range(y[0].start_time, y[-1].start_time).to_period(freq)
    num_freq = (x[-1] - x[0]).n
    return num_freq / num_seasonality


def seasonalities_to_array(
    seasonalities: Sequence[float | str],
    freq: str,
) -> np.ndarray:
    """Convert a list of floats or strings to durations relative to a frequency."""
    ret = []
    for seasonality in seasonalities:
        if isinstance(seasonality, str):
            seasonality_float = seasonality_to_float(seasonality, freq)
            if seasonality_float < 1:
                raise TypeError(
                    f"{seasonality=} should represent a time span greater than {freq=}, "
                    f"but {seasonality} is {seasonality_float:.2f} of a {freq}"
                )
        else:
            seasonality_float = seasonality
            if seasonality_float < 1:
                raise TypeError(f"{seasonality_float=} should be larger than 1.")
        ret.append(seasonality_float)
    return np.asarray(ret, dtype=np.float32)


def _convert_datetime_col(table, time_column, timetype, freq, time_min=None):
    """Converts a time column in place according to the frequency."""
    if timetype == "index":
        first_date = pd.to_datetime("2020-01-01").to_period(freq)
        table[time_column] = pd.to_datetime(table[time_column], errors="coerce")
        if table[time_column].isna().any():
            num_bad = int(table[time_column].isna().sum())
            raise ValueError(
                f"Column `{time_column}` contains {num_bad} invalid datetimes after parsing."
            )
        table[time_column] = table[time_column].dt.to_period(freq)
        table[time_column] = (table[time_column] - first_date).apply(lambda x: x.n)
    elif timetype == "float":
        table[time_column] = pd.to_numeric(table[time_column], errors="coerce")
        if table[time_column].isna().any():
            num_bad = int(table[time_column].isna().sum())
            raise ValueError(
                f"Column `{time_column}` contains {num_bad} invalid numeric values."
            )
    else:
        raise ValueError(f"Unknown timetype: {timetype}")

    if time_min is None:
        time_min = table[time_column].min()
    table[time_column] = table[time_column] - time_min
    return table, time_min


def _safe_nanmedian(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    if values.size == 0 or np.isnan(values).all():
        return float(default)
    return float(np.nanmedian(values))


def _quantile_label(q: float) -> str:
    return f"{int(round(float(q) * 1000.0)):03d}"


def _normalize_coverage_levels(
    coverage_levels: Sequence[float],
) -> tuple[float, ...]:
    normalized = []
    for level in coverage_levels:
        level = float(level)
        if not 0.0 < level < 1.0:
            raise ValueError(f"Coverage level must be in (0, 1), but received {level}.")
        if level not in normalized:
            normalized.append(level)
    return tuple(normalized)


def _canonical_quantile_column(base_name: str, q: float) -> str:
    return f"{base_name}_q{_quantile_label(q)}"


def _legacy_quantile_column(base_name: str, q: float) -> str:
    return f"{base_name}_q{q:.3f}".replace(".", "")


def _find_quantile_column(
    table: pd.DataFrame,
    base_name: str,
    q: float,
) -> str | None:
    candidates = [
        _canonical_quantile_column(base_name, q),
        _legacy_quantile_column(base_name, q),
    ]
    if abs(float(q) - 0.025) < 1e-9:
        candidates.append(f"{base_name}_lower")
    if abs(float(q) - 0.975) < 1e-9:
        candidates.append(f"{base_name}_upper")
    for candidate in candidates:
        if candidate in table.columns:
            return candidate
    return None


def _reduce_ensemble_mean(ensemble_means: np.ndarray) -> np.ndarray:
    ensemble_means = np.asarray(ensemble_means, dtype=np.float64)
    if ensemble_means.ndim == 0:
        raise ValueError("ensemble_means must contain a sample axis.")
    if ensemble_means.ndim == 1:
        return ensemble_means.astype(np.float32)
    ensemble_axes = tuple(range(ensemble_means.ndim - 1))
    return ensemble_means.mean(axis=ensemble_axes).astype(np.float32)


def _compute_uncertainty_decomposition(
    ensemble_means: np.ndarray,
    ensemble_scales: np.ndarray,
    eps: float = EPS,
    min_std: float = MIN_STD,
) -> dict[str, np.ndarray]:
    """Decompose predictive uncertainty from ensemble means and scales."""
    ensemble_means = np.asarray(ensemble_means, dtype=np.float64)
    ensemble_scales = np.asarray(ensemble_scales, dtype=np.float64)
    if ensemble_means.shape != ensemble_scales.shape:
        raise ValueError(
            "ensemble_means and ensemble_scales must have identical shapes, "
            f"got {ensemble_means.shape} and {ensemble_scales.shape}."
        )
    if ensemble_means.ndim < 2:
        raise ValueError(
            "Expected ensemble outputs with one or more ensemble axes plus a "
            f"sample axis, but received shape {ensemble_means.shape}."
        )

    ensemble_axes = tuple(range(ensemble_means.ndim - 1))
    safe_scales = np.maximum(ensemble_scales, min_std)
    epistemic_var = np.var(ensemble_means, axis=ensemble_axes)
    aleatoric_var = np.mean(np.square(safe_scales), axis=ensemble_axes)
    total_var = np.maximum(epistemic_var + aleatoric_var, 0.0)

    epistemic_std = np.sqrt(np.maximum(epistemic_var, 0.0))
    aleatoric_std = np.sqrt(np.maximum(aleatoric_var, 0.0))
    total_std = np.sqrt(total_var)

    denom = total_var + eps
    epistemic_ratio = np.clip(epistemic_var / denom, 0.0, 1.0)
    aleatoric_ratio = np.clip(aleatoric_var / denom, 0.0, 1.0)

    return {
        "epistemic_var": epistemic_var.astype(np.float32),
        "aleatoric_var": aleatoric_var.astype(np.float32),
        "total_var": total_var.astype(np.float32),
        "epistemic_std": epistemic_std.astype(np.float32),
        "aleatoric_std": aleatoric_std.astype(np.float32),
        "total_std": total_std.astype(np.float32),
        "epistemic_ratio": epistemic_ratio.astype(np.float32),
        "aleatoric_ratio": aleatoric_ratio.astype(np.float32),
    }


def _write_uncertainty_columns(
    result: pd.DataFrame,
    base_name: str,
    decomposition: dict[str, np.ndarray],
) -> None:
    for suffix, values in decomposition.items():
        result[f"{base_name}_{suffix}"] = np.asarray(values, dtype=np.float32)


def _uncertainty_summary_metrics(
    result: pd.DataFrame,
    prediction_col: str,
) -> dict[str, float]:
    """Summarize uncertainty magnitudes for heteroscedastic diagnostics."""
    metrics = {}
    for suffix in (
        "epistemic_var",
        "aleatoric_var",
        "total_var",
        "epistemic_std",
        "aleatoric_std",
        "total_std",
        "epistemic_ratio",
        "aleatoric_ratio",
    ):
        col = f"{prediction_col}_{suffix}"
        if col not in result.columns:
            continue
        values = pd.to_numeric(result[col], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        if int(mask.sum()) == 0:
            metrics[f"{suffix}_mean"] = float("nan")
            metrics[f"{suffix}_median"] = float("nan")
            continue
        values = values[mask]
        metrics[f"{suffix}_mean"] = float(np.mean(values))
        metrics[f"{suffix}_median"] = float(np.median(values))
    return metrics


def _smape_value(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denominator))


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) < EPS or np.nanstd(y) < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _gaussian_nll(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    min_std: float = MIN_STD,
) -> np.ndarray:
    y_pred_std = np.maximum(np.asarray(y_pred_std, dtype=np.float64), min_std)
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(
        y_pred_mean, dtype=np.float64
    )
    return 0.5 * np.log(2.0 * np.pi * np.square(y_pred_std)) + 0.5 * np.square(
        residual / y_pred_std
    )


def _predictive_mean_column(result: pd.DataFrame, prediction_col: str) -> str:
    ensemble_mean_col = f"{prediction_col}_ensemble_mean"
    return ensemble_mean_col if ensemble_mean_col in result.columns else prediction_col


def _extract_evaluation_arrays(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
) -> dict[str, np.ndarray]:
    total_std_col = f"{prediction_col}_total_std"
    total_var_col = f"{prediction_col}_total_var"
    if total_std_col not in result.columns:
        raise ValueError(
            f"Cannot evaluate uncertainty because `{total_std_col}` is missing."
        )

    predictive_mean_col = _predictive_mean_column(result, prediction_col)
    y_true = pd.to_numeric(result[target_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    y_pred = pd.to_numeric(result[prediction_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    predictive_mean = pd.to_numeric(
        result[predictive_mean_col], errors="coerce"
    ).to_numpy(dtype=np.float64)
    total_std = pd.to_numeric(result[total_std_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if total_var_col in result.columns:
        total_var = pd.to_numeric(result[total_var_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        total_var = np.square(total_std)

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(predictive_mean)
        & np.isfinite(total_std)
        & np.isfinite(total_var)
    )
    return {
        "mask": mask,
        "y_true": y_true,
        "y_pred": y_pred,
        "predictive_mean": predictive_mean,
        "total_std": total_std,
        "total_var": total_var,
    }


def _get_prediction_interval(
    result: pd.DataFrame,
    prediction_col: str,
    coverage_level: float,
    predictive_mean: np.ndarray,
    total_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - float(coverage_level)) / 2.0
    lower_col = _find_quantile_column(result, prediction_col, alpha)
    upper_col = _find_quantile_column(result, prediction_col, 1.0 - alpha)
    if lower_col is not None and upper_col is not None:
        lower = pd.to_numeric(result[lower_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        upper = pd.to_numeric(result[upper_col], errors="coerce").to_numpy(
            dtype=np.float64
        )
        return lower, upper

    safe_std = np.maximum(np.asarray(total_std, dtype=np.float64), MIN_STD)
    quantile = min(max(0.5 + float(coverage_level) / 2.0, EPS), 1.0 - EPS)
    z_value = NormalDist().inv_cdf(quantile)
    predictive_mean = np.asarray(predictive_mean, dtype=np.float64)
    return predictive_mean - z_value * safe_std, predictive_mean + z_value * safe_std


def _build_reliability_curve(
    predicted_std: np.ndarray,
    predicted_var: np.ndarray,
    squared_error: np.ndarray,
    num_bins: int = DEFAULT_CALIBRATION_BINS,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    predicted_std = np.asarray(predicted_std, dtype=np.float64)
    predicted_var = np.asarray(predicted_var, dtype=np.float64)
    squared_error = np.asarray(squared_error, dtype=np.float64)
    mask = (
        np.isfinite(predicted_std)
        & np.isfinite(predicted_var)
        & np.isfinite(squared_error)
    )
    if int(mask.sum()) == 0:
        empty = np.zeros((0,), dtype=np.float64)
        return {"rmv": empty, "rmse": empty, "count": empty}

    predicted_std = predicted_std[mask]
    predicted_var = np.maximum(predicted_var[mask], 0.0)
    squared_error = np.maximum(squared_error[mask], 0.0)

    num_bins = max(1, min(int(num_bins), predicted_std.shape[0]))
    order = np.argsort(predicted_std)
    bin_indices = np.array_split(order, num_bins)

    rmv = []
    rmse = []
    counts = []
    for idx in bin_indices:
        if idx.size == 0:
            continue
        rmv.append(np.sqrt(max(float(np.mean(predicted_var[idx])), 0.0)))
        rmse.append(np.sqrt(max(float(np.mean(squared_error[idx])), 0.0)))
        counts.append(float(idx.size))

    return {
        "rmv": np.asarray(rmv, dtype=np.float64),
        "rmse": np.asarray(rmse, dtype=np.float64),
        "count": np.asarray(counts, dtype=np.float64),
    }


def build_reliability_diagram_data(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    num_bins: int = DEFAULT_CALIBRATION_BINS,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    arrays = _extract_evaluation_arrays(result, prediction_col, target_col)
    mask = arrays["mask"]
    squared_error = np.square(arrays["y_true"][mask] - arrays["y_pred"][mask])
    return _build_reliability_curve(
        arrays["total_std"][mask],
        arrays["total_var"][mask],
        squared_error,
        num_bins=num_bins,
        eps=eps,
    )


def build_coverage_plot_data(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> dict[str, np.ndarray]:
    coverage_levels = _normalize_coverage_levels(coverage_levels)
    arrays = _extract_evaluation_arrays(result, prediction_col, target_col)
    mask = arrays["mask"]

    empirical_coverages = []
    mpiws = []
    for coverage_level in coverage_levels:
        lower, upper = _get_prediction_interval(
            result,
            prediction_col,
            coverage_level,
            arrays["predictive_mean"],
            arrays["total_std"],
        )
        interval_mask = mask & np.isfinite(lower) & np.isfinite(upper)
        if int(interval_mask.sum()) == 0:
            empirical_coverages.append(np.nan)
            mpiws.append(np.nan)
            continue
        y_true = arrays["y_true"][interval_mask]
        lower = lower[interval_mask]
        upper = upper[interval_mask]
        empirical_coverages.append(
            float(np.mean((y_true >= lower) & (y_true <= upper)))
        )
        mpiws.append(float(np.mean(upper - lower)))

    return {
        "nominal": np.asarray(coverage_levels, dtype=np.float64),
        "empirical": np.asarray(empirical_coverages, dtype=np.float64),
        "mpiw": np.asarray(mpiws, dtype=np.float64),
    }


def compute_calibration_metrics(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, float]:
    coverage_levels = _normalize_coverage_levels(coverage_levels)
    arrays = _extract_evaluation_arrays(result, prediction_col, target_col)
    mask = arrays["mask"]

    if int(mask.sum()) == 0:
        return {
            "nll": float("nan"),
            "mace": float("nan"),
            "ence": float("nan"),
            "error_std_corr": float("nan"),
            "error_var_corr": float("nan"),
        }

    y_true = arrays["y_true"][mask]
    y_pred = arrays["y_pred"][mask]
    predictive_mean = arrays["predictive_mean"][mask]
    total_std = np.maximum(arrays["total_std"][mask], MIN_STD)
    total_var = np.maximum(arrays["total_var"][mask], 0.0)

    absolute_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    metrics = {
        "nll": float(np.mean(_gaussian_nll(y_true, predictive_mean, total_std))),
        "absolute_error_mean": float(np.mean(absolute_error)),
        "squared_error_mean": float(np.mean(squared_error)),
    }

    calibration_curve = build_coverage_plot_data(
        result,
        prediction_col,
        target_col,
        coverage_levels=coverage_levels,
    )
    mace_terms = []
    for coverage_level, empirical, mpiw in zip(
        calibration_curve["nominal"],
        calibration_curve["empirical"],
        calibration_curve["mpiw"],
    ):
        label = int(round(float(coverage_level) * 100.0))
        metrics[f"picp_{label}"] = (
            float(empirical) if np.isfinite(empirical) else float("nan")
        )
        metrics[f"mpiw_{label}"] = float(mpiw) if np.isfinite(mpiw) else float("nan")
        if np.isfinite(empirical):
            mace_terms.append(abs(float(empirical) - float(coverage_level)))
    metrics["mace"] = float(np.mean(mace_terms)) if mace_terms else float("nan")
    metrics["ece"] = metrics["mace"]

    reliability = _build_reliability_curve(
        total_std,
        total_var,
        squared_error,
        num_bins=calibration_bins,
    )
    if reliability["rmv"].size > 0:
        metrics["ence"] = float(
            np.mean(
                np.abs(reliability["rmse"] - reliability["rmv"])
                / np.maximum(reliability["rmv"], EPS)
            )
        )
        metrics["reliability_bins"] = int(reliability["rmv"].size)
    else:
        metrics["ence"] = float("nan")
        metrics["reliability_bins"] = 0

    metrics["abs_error_std_corr"] = _safe_correlation(absolute_error, total_std)
    metrics["abs_error_var_corr"] = _safe_correlation(absolute_error, total_var)
    metrics["sq_error_std_corr"] = _safe_correlation(squared_error, total_std)
    metrics["sq_error_var_corr"] = _safe_correlation(squared_error, total_var)
    metrics["error_std_corr"] = metrics["abs_error_std_corr"]
    metrics["error_var_corr"] = metrics["sq_error_var_corr"]
    return metrics


def plot_reliability_diagram(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    save_path: str | Path,
    num_bins: int = DEFAULT_CALIBRATION_BINS,
) -> Path:
    from matplotlib import pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_reliability_diagram_data(
        result,
        prediction_col,
        target_col,
        num_bins=num_bins,
    )

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    if data["rmv"].size > 0:
        max_axis = float(max(np.max(data["rmv"]), np.max(data["rmse"]), MIN_STD))
        ax.scatter(data["rmv"], data["rmse"], alpha=0.8, s=40)
    else:
        max_axis = 1.0
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
    ax.plot([0.0, max_axis], [0.0, max_axis], "k--", linewidth=1.0)
    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")
    ax.set_title("Reliability Diagram")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_coverage_diagram(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    save_path: str | Path,
    coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> Path:
    from matplotlib import pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_coverage_plot_data(
        result,
        prediction_col,
        target_col,
        coverage_levels=coverage_levels,
    )

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    ax.plot(data["nominal"], data["empirical"], marker="o")
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1.0)
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Coverage Plot")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_uncertainty_error_scatter(
    result: pd.DataFrame,
    prediction_col: str,
    target_col: str,
    save_path: str | Path,
    error_kind: str = "absolute_error",
    max_points: int = 5000,
    random_seed: int = 0,
) -> Path:
    from matplotlib import pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _extract_evaluation_arrays(result, prediction_col, target_col)
    mask = arrays["mask"]
    predicted_std = np.maximum(arrays["total_std"][mask], MIN_STD)

    residual = arrays["y_true"][mask] - arrays["y_pred"][mask]
    if error_kind == "squared_error":
        error_values = np.square(residual)
        ylabel = "Squared Error"
    else:
        error_values = np.abs(residual)
        ylabel = "Absolute Error"

    if predicted_std.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(predicted_std.shape[0], size=max_points, replace=False)
        predicted_std = predicted_std[selected]
        error_values = error_values[selected]

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    if predicted_std.size > 0:
        ax.scatter(predicted_std, error_values, alpha=0.35, s=10)
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
    ax.set_xlabel("Predicted Total Std")
    ax.set_ylabel(ylabel)
    ax.set_title("Uncertainty vs Error")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def split_metrics_for_export(
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_keys = {
        "mae",
        "rmse",
        "r2",
        "smape",
        "n_eval",
        "n_total",
        "target_col",
        "model_target_col",
        "prediction_col",
        "residual_base_col",
        "prediction_mode",
        "n_base_missing",
    }
    summary_metrics = {k: metrics[k] for k in summary_keys if k in metrics}
    calibration_metrics = {k: v for k, v in metrics.items() if k not in summary_keys}
    return summary_metrics, calibration_metrics


class ReferenceGATLayer(nn.Module):
    """Small GAT used only once to encode reference stations."""

    output_dim: int = 16
    head_dim: int = 4
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, adj):
        x = jnp.asarray(x, dtype=jnp.float32)
        adj = jnp.asarray(adj, dtype=jnp.float32)

        x_proj = nn.Dense(self.head_dim * self.num_heads)(x)
        x_proj = x_proj.reshape((x.shape[0], self.num_heads, self.head_dim))

        scores = jnp.einsum("ihd,jhd->ijh", x_proj, x_proj)
        scores = scores / jnp.sqrt(float(self.head_dim))

        mask = adj > 0
        scores = jnp.where(
            mask[..., None], scores + jnp.log(adj[..., None] + 1e-6), -1e9
        )
        attn = nn.softmax(scores, axis=1)

        h = jnp.einsum("ijh,jhd->ihd", attn, x_proj)
        h = h.reshape((x.shape[0], self.head_dim * self.num_heads))
        return nn.Dense(self.output_dim)(h)


class SpatiotemporalDataHandler:
    """Base class for preparing spatiotemporal data.

    Key design choices for sparse mobile monitoring:
    - supervised fitting still uses rows with observed `target_col` only;
    - train-set feature imputation / scaling statistics are learned once and then
      reused on validation / test to guarantee consistency;
    - spatial reference points are built from *all* available training rows using
      grid cells (if present) or rounded coordinates, instead of exact raw
      coordinates, which are usually almost unique for moving trajectories.
    """

    def __init__(
        self,
        feature_cols: Sequence[str],
        target_col: str,
        timetype: str,
        freq: str,
        standardize: Sequence[str] | None = None,
        max_ref_points: int = 256,
        ref_knn: int = 8,
        ref_feature_dim: int = 16,
        reference_group_col: str | None = "grid",
        reference_coord_round: int = 3,
        use_all_train_for_reference: bool = True,
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.timetype = timetype
        self.freq = freq
        self.standardize = list(standardize) if standardize is not None else None
        self.max_ref_points = int(max_ref_points)
        self.ref_knn = int(ref_knn)
        self.ref_feature_dim = int(ref_feature_dim)
        self.reference_group_col = reference_group_col
        self.reference_coord_round = int(reference_coord_round)
        self.use_all_train_for_reference = bool(use_all_train_for_reference)

        self.mu_ = None
        self.std_ = None
        self.time_min_ = None
        self.time_scale_ = None
        self.ref_points = None
        self.feature_fill_values_ = None

    @property
    def _time_idx(self) -> int:
        return 0

    @property
    def _time_column(self) -> str:
        return self.feature_cols[self._time_idx]

    def _validate_columns(
        self, table: pd.DataFrame, require_target: bool
    ) -> pd.DataFrame:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("Input table must be a pandas DataFrame.")
        if table.empty:
            raise ValueError("Input table is empty.")

        required_cols = list(self.feature_cols)
        if require_target:
            required_cols.append(self.target_col)
        missing = [c for c in required_cols if c not in table.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        table = table.copy()
        for col in self.feature_cols:
            if col == self._time_column:
                continue
            table[col] = pd.to_numeric(table[col], errors="coerce")

        if self.target_col in table.columns:
            table[self.target_col] = pd.to_numeric(
                table[self.target_col], errors="coerce"
            )
        return table

    def get_target(self, table: pd.DataFrame) -> np.ndarray:
        table = self._validate_columns(table, require_target=True)
        table = self._maybe_filter_target_nans(table)
        return table[self.target_col].to_numpy(dtype=np.float32)

    def _maybe_filter_target_nans(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.target_col in table.columns:
            return table[table[self.target_col].notna()].copy()
        return table.copy()

    def copy_and_filter_table(self, table: pd.DataFrame) -> pd.DataFrame:
        table = self._validate_columns(table, require_target=True)
        return self._maybe_filter_target_nans(table)

    def get_reference_points(self) -> jnp.ndarray:
        if self.ref_points is None:
            return jnp.zeros((0, 2 + self.ref_feature_dim), dtype=jnp.float32)
        return jnp.asarray(self.ref_points, dtype=jnp.float32)

    def _fit_feature_fill_values(self, table: pd.DataFrame) -> None:
        fill_values = {}
        for col in self.feature_cols:
            if col == self._time_column:
                continue
            fill_values[col] = _safe_nanmedian(table[col], default=0.0)
        self.feature_fill_values_ = fill_values

    def _fill_missing_features(self, table: pd.DataFrame) -> pd.DataFrame:
        if self.feature_fill_values_ is None:
            raise ValueError(
                "Feature fill values are not initialized. Call `get_train` first."
            )
        table = table.copy()
        for col, fill_value in self.feature_fill_values_.items():
            table[col] = table[col].fillna(fill_value)
        return table

    def _fit_standardization_stats(self, features: np.ndarray) -> None:
        self.mu_ = np.zeros(len(self.feature_cols), dtype=np.float32)
        self.std_ = np.ones(len(self.feature_cols), dtype=np.float32)

        if not self.standardize:
            return

        if self._time_column in self.standardize:
            raise TypeError("Do not standardize the time column!")

        idx = [self.feature_cols.index(f) for f in self.standardize]
        self.mu_[idx] = np.nanmean(features[:, idx].astype(np.float32), axis=0)
        self.std_[idx] = np.nanstd(features[:, idx].astype(np.float32), axis=0)
        self.std_[idx] = np.where(self.std_[idx] < 1e-6, 1.0, self.std_[idx])

    def _apply_standardization(self, features: np.ndarray) -> np.ndarray:
        features = features.astype(np.float32, copy=False)
        if self.standardize:
            idx = [self.feature_cols.index(f) for f in self.standardize]
            features[:, idx] = (features[:, idx] - self.mu_[idx]) / self.std_[idx]
        return features.astype(np.float32)

    def _scale_columns(self, values: np.ndarray, cols: Sequence[str]) -> np.ndarray:
        values = values.astype(np.float32, copy=True)
        if not cols:
            return values
        for i, col in enumerate(cols):
            if self.standardize and col in self.standardize:
                col_idx = self.feature_cols.index(col)
                values[:, i] = (values[:, i] - self.mu_[col_idx]) / self.std_[col_idx]
        return values

    def _reference_group_keys(
        self, table: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        table = table.copy()
        if (
            self.reference_group_col is not None
            and self.reference_group_col in table.columns
            and table[self.reference_group_col].notna().any()
        ):
            return table, [self.reference_group_col]

        table["_ref_lat_bin"] = table["latitude"].round(self.reference_coord_round)
        table["_ref_lon_bin"] = table["longitude"].round(self.reference_coord_round)
        return table, ["_ref_lat_bin", "_ref_lon_bin"]

    def _build_reference_points(self, table: pd.DataFrame) -> None:
        """Build compact reference points for spatial features.

        This version is intentionally robust to mobile trajectories:
        - uses train rows after feature imputation, so unlabeled train rows can still
          contribute spatial coverage without leaking the target;
        - groups by `grid` when available, otherwise by rounded coordinates, which is
          much more stable than exact coordinate matching for moving taxis.
        """
        coord_cols = ["latitude", "longitude"]
        if not set(coord_cols).issubset(table.columns):
            self.ref_points = None
            return

        table = table.copy()
        table, group_keys = self._reference_group_keys(table)

        agg_feature_cols = [
            c
            for c in self.feature_cols
            if c not in (self._time_column, "latitude", "longitude")
            and c not in group_keys
        ]
        if len(agg_feature_cols) == 0:
            agg_feature_cols = ["latitude", "longitude"]

        agg_dict: dict[str, Any] = {"latitude": "mean", "longitude": "mean"}
        for col in agg_feature_cols:
            agg_dict[col] = "mean"

        grouped = table.groupby(group_keys, dropna=False, as_index=False).agg(agg_dict)
        counts = (
            table.groupby(group_keys, dropna=False)
            .size()
            .reset_index(name="sample_count")
        )
        grouped = (
            grouped.merge(counts, on=group_keys, how="left")
            .sort_values("sample_count", ascending=False)
            .reset_index(drop=True)
        )

        if grouped.empty:
            self.ref_points = None
            return

        grouped = grouped.head(self.max_ref_points).copy()

        coords_raw = grouped[["latitude", "longitude"]].to_numpy(dtype=np.float32)
        coords_scaled = self._scale_columns(coords_raw, ["latitude", "longitude"])

        node_features_raw = grouped[agg_feature_cols].to_numpy(dtype=np.float32)
        node_features = self._scale_columns(node_features_raw, agg_feature_cols)

        num_nodes = coords_scaled.shape[0]
        if num_nodes == 1:
            adj = np.ones((1, 1), dtype=np.float32)
        else:
            knn = min(self.ref_knn + 1, num_nodes)
            nbrs = NearestNeighbors(n_neighbors=knn, algorithm="auto")
            nbrs.fit(coords_scaled)
            distances, indices = nbrs.kneighbors(coords_scaled)

            positive_dist = distances[:, 1:] if distances.shape[1] > 1 else distances
            valid = positive_dist[positive_dist > 0]
            sigma = float(np.median(valid)) if valid.size > 0 else 1.0
            sigma = max(sigma, 1e-6)

            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for i in range(num_nodes):
                for d, j in zip(distances[i], indices[i]):
                    weight = np.exp(-(d**2) / (2.0 * sigma**2))
                    adj[i, j] = max(adj[i, j], weight)
                    adj[j, i] = max(adj[j, i], weight)
            np.fill_diagonal(adj, 1.0)

        gat = ReferenceGATLayer(output_dim=self.ref_feature_dim)
        key = jax.random.PRNGKey(0)
        params = gat.init(
            key,
            jnp.asarray(node_features, dtype=jnp.float32),
            jnp.asarray(adj, dtype=jnp.float32),
        )
        gat_features = np.asarray(
            gat.apply(
                params,
                jnp.asarray(node_features, dtype=jnp.float32),
                jnp.asarray(adj, dtype=jnp.float32),
            )
        )

        self.ref_points = np.concatenate(
            [coords_scaled.astype(np.float32), gat_features.astype(np.float32)],
            axis=1,
        )

    def get_train(self, table: pd.DataFrame) -> np.ndarray:
        """Fetch supervised training data.

        The model is still trained on rows with observed targets only, but the train
        statistics for imputation / scaling / spatial anchors are learned from the
        whole training split to make better use of sparse-label mobile data.
        """
        full_table = self._validate_columns(table, require_target=False)
        supervised_table = self.copy_and_filter_table(table)

        if supervised_table.empty:
            raise ValueError(
                f"No non-null `{self.target_col}` rows were found in the training data."
            )

        full_table, self.time_min_ = _convert_datetime_col(
            full_table, self._time_column, self.timetype, self.freq, None
        )
        supervised_table, _ = _convert_datetime_col(
            supervised_table,
            self._time_column,
            self.timetype,
            self.freq,
            self.time_min_,
        )

        self._fit_feature_fill_values(full_table)
        full_table = self._fill_missing_features(full_table)
        supervised_table = self._fill_missing_features(supervised_table)

        full_features = full_table[self.feature_cols].to_numpy(dtype=np.float32)
        supervised_features = supervised_table[self.feature_cols].to_numpy(
            dtype=np.float32
        )

        self.time_scale_ = float(max(full_features[:, self._time_idx].max(), 1.0))
        self._fit_standardization_stats(full_features)

        full_features = self._apply_standardization(full_features)
        supervised_features = self._apply_standardization(supervised_features)

        reference_source = (
            full_table if self.use_all_train_for_reference else supervised_table
        )
        self._build_reference_points(reference_source)
        return supervised_features.astype(np.float32)

    def get_test(self, table: pd.DataFrame) -> np.ndarray:
        """Fetch validation/test data. Call this after `get_train`."""
        if self.time_min_ is None:
            raise ValueError("The data handler is not fitted. Call `get_train` first.")

        table = self._validate_columns(table, require_target=False)
        table, _ = _convert_datetime_col(
            table, self._time_column, self.timetype, self.freq, self.time_min_
        )
        table = self._fill_missing_features(table)

        features = table[self.feature_cols].to_numpy(dtype=np.float32)
        features = self._apply_standardization(features)

        if np.isnan(features).any():
            raise ValueError("NaNs remain in transformed features after preprocessing.")
        return features.astype(np.float32)

    def get_input_scales(self) -> np.ndarray:
        input_scales = np.ones(len(self.feature_cols), dtype=np.float32)
        input_scales[self._time_idx] = max(float(self.time_scale_), 1.0)
        return input_scales


class BayesianNeuralFieldEstimator:
    """Base estimator for spatiotemporal neural-field models."""

    _ensemble_dims: int
    _prior_weight: float = 1.0
    _scale_epochs_by_batch_size: bool = False

    def __init__(
        self,
        *,
        feature_cols: Sequence[str],
        target_col: str,
        seasonality_periods: Sequence[float | str] | None = None,
        num_seasonal_harmonics: Sequence[int] | None = None,
        fourier_degrees: Sequence[float] | None = None,
        interactions: Sequence[tuple[int, int]] | None = None,
        freq: str | None = None,
        timetype: str = "index",
        depth: int = 2,
        width: int = 512,
        observation_model: str = "NORMAL",
        standardize: Sequence[str] | None = None,
        max_ref_points: int = 256,
        ref_knn: int = 8,
        ref_feature_dim: int = 16,
        use_remat: bool = True,
        reference_group_col: str | None = "grid",
        reference_coord_round: int = 3,
        use_all_train_for_reference: bool = True,
        residual_base_col: str | None = None,
        evaluation_target_col: str | None = None,
        min_log_variance: float = -8.0,
        max_log_variance: float = 6.0,
        variance_output_init_scale: float = 0.05,
        detach_variance_head: bool = True,
        variance_regularization_weight: float = 1e-4,
        variance_reference_scale_factor: float = 0.5,
    ):
        self.num_seasonal_harmonics = num_seasonal_harmonics
        self.seasonality_periods = seasonality_periods
        self.observation_model = observation_model
        self.depth = depth
        self.width = width
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.timetype = timetype
        self.freq = freq
        self.fourier_degrees = fourier_degrees
        self.standardize = list(standardize) if standardize is not None else None
        self.interactions = interactions
        self.use_remat = use_remat
        self.residual_base_col = residual_base_col
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self.variance_output_init_scale = float(variance_output_init_scale)
        self.detach_variance_head = bool(detach_variance_head)
        self.variance_regularization_weight = float(variance_regularization_weight)
        self.variance_reference_scale_factor = float(variance_reference_scale_factor)
        self.evaluation_target_col = (
            evaluation_target_col if evaluation_target_col is not None else target_col
        )

        self.losses_ = None
        self.params_ = None
        self.data_handler = SpatiotemporalDataHandler(
            self.feature_cols,
            self.target_col,
            self.timetype,
            self.freq,
            standardize=self.standardize,
            max_ref_points=max_ref_points,
            ref_knn=ref_knn,
            ref_feature_dim=ref_feature_dim,
            reference_group_col=reference_group_col,
            reference_coord_round=reference_coord_round,
            use_all_train_for_reference=use_all_train_for_reference,
        )

    def _get_fourier_degrees(self, batch_shape: tuple[int, ...]) -> np.ndarray:
        if self.fourier_degrees is None:
            fourier_degrees = np.full(batch_shape[-1], 5, dtype=int)
        else:
            fourier_degrees = np.atleast_1d(self.fourier_degrees).astype(int)
            if fourier_degrees.shape[-1] != batch_shape[-1]:
                raise ValueError(
                    "The length of fourier_degrees ({}) must match the input "
                    "dimension dimension ({}).".format(
                        fourier_degrees.shape[-1], batch_shape[-1]
                    )
                )
        return fourier_degrees

    def _get_interactions(self) -> np.ndarray:
        if self.interactions is None:
            interactions = np.zeros((0, 2), dtype=int)
        else:
            interactions = np.asarray(self.interactions).astype(int)
            if np.ndim(interactions) != 2 or interactions.shape[-1] != 2:
                raise ValueError(
                    "The argument for `interactions` should be a 2-d array of integers "
                    "of shape (N, 2), indicating the column indices to interact "
                    f"(the passed shape was {interactions.shape})"
                )
        return interactions

    def _get_seasonality_periods(self):
        if (self.timetype == "index" and self.freq is None) or (
            self.timetype == "float" and self.freq is not None
        ):
            raise ValueError(f"Invalid {self.freq=} with {self.timetype=}.")
        if self.seasonality_periods is None:
            return np.zeros(0, dtype=np.float32)
        if self.timetype == "index":
            return seasonalities_to_array(self.seasonality_periods, self.freq)
        if self.timetype == "float":
            return np.asarray(self.seasonality_periods, dtype=np.float32)
        assert False, f"Impossible {self.timetype=}."

    def _get_num_seasonal_harmonics(self):
        if self.timetype == "index":
            return (
                np.asarray(self.num_seasonal_harmonics, dtype=np.float32)
                if self.num_seasonal_harmonics is not None
                else np.zeros(0, dtype=np.float32)
            )
        if self.timetype == "float":
            if self.num_seasonal_harmonics is not None:
                raise ValueError(
                    f"Cannot use num_seasonal_harmonics with {self.timetype=}."
                )
            return np.fmin(0.5, self._get_seasonality_periods() / 2)
        assert False, f"Impossible {self.timetype=}."

    def _model_args(self, batch_shape):
        feature_dim = int(batch_shape[-1])
        return {
            "depth": self.depth,
            "input_scales": self.data_handler.get_input_scales(),
            "num_seasonal_harmonics": self._get_num_seasonal_harmonics(),
            "seasonality_periods": self._get_seasonality_periods(),
            "width": self.width,
            "init_x": (1, feature_dim),
            "fourier_degrees": self._get_fourier_degrees((1, feature_dim)),
            "interactions": self._get_interactions(),
            "ref_points": self.data_handler.get_reference_points(),
            "spatial_k": self.data_handler.ref_knn,
            "use_remat": self.use_remat,
            "min_log_variance": self.min_log_variance,
            "max_log_variance": self.max_log_variance,
            "variance_output_init_scale": self.variance_output_init_scale,
            "detach_variance_head": self.detach_variance_head,
        }

    def predict(
        self,
        table,
        quantiles=(0.5,),
        approximate_quantiles=False,
        prediction_batch_size: int = 2048,
        keep_ensemble_means: bool = True,
        keep_ensemble_scales: bool = False,
        return_aux: bool = False,
    ):
        """Make predictions of the target column at new times."""
        if self.params_ is None:
            raise ValueError("The model is not fitted yet. Call `fit` first.")
        test_data = self.data_handler.get_test(table)
        return inference.predict_bnf(
            test_data,
            self.observation_model,
            params=self.params_,
            model_args=self._model_args(test_data.shape),
            quantiles=quantiles,
            ensemble_dims=self._ensemble_dims,
            approximate_quantiles=approximate_quantiles,
            batch_size=prediction_batch_size,
            keep_ensemble_means=keep_ensemble_means,
            keep_ensemble_scales=keep_ensemble_scales,
            return_aux=return_aux,
        )

    def predict_dataframe(
        self,
        table: pd.DataFrame,
        quantiles=(0.025, 0.5, 0.975),
        prediction_col: str = "predict_pm10",
        approximate_quantiles: bool = False,
        prediction_batch_size: int = 2048,
        keep_ensemble_means: bool = False,
    ) -> pd.DataFrame:
        """Return a copy of `table` with prediction and uncertainty columns."""
        result = table.copy()
        request_ensemble_mean_columns = bool(keep_ensemble_means)
        means, pred_quantiles, predict_aux = self.predict(
            table,
            quantiles=quantiles,
            approximate_quantiles=approximate_quantiles,
            prediction_batch_size=prediction_batch_size,
            keep_ensemble_means=True,
            keep_ensemble_scales=True,
            return_aux=True,
        )
        ensemble_means = (
            predict_aux.get("ensemble_means") if predict_aux is not None else means
        )
        ensemble_scales = (
            predict_aux.get("ensemble_scales") if predict_aux is not None else None
        )
        if ensemble_means is None or ensemble_scales is None:
            raise ValueError(
                "Uncertainty decomposition requires ensemble means and scales."
            )

        pred_quantiles = [
            np.asarray(q).reshape(-1).astype(np.float32) for q in pred_quantiles
        ]
        if any(len(q) != len(result) for q in pred_quantiles):
            lengths = [len(q) for q in pred_quantiles]
            raise ValueError(
                f"Prediction length mismatch. table={len(result)}, quantiles={lengths}"
            )

        quantile_map = {float(q): arr for q, arr in zip(quantiles, pred_quantiles)}

        def _write_quantile_columns(
            base_name: str, values_map: dict[float, np.ndarray]
        ) -> None:
            for q, values in values_map.items():
                canonical = _canonical_quantile_column(base_name, q)
                legacy = _legacy_quantile_column(base_name, q)
                result[canonical] = values
                if legacy != canonical:
                    result[legacy] = values

        mean_arr = _reduce_ensemble_mean(ensemble_means)
        if len(mean_arr) != len(result):
            raise ValueError(
                f"Ensemble mean length mismatch. table={len(result)}, means={len(mean_arr)}"
            )

        decomposition = _compute_uncertainty_decomposition(
            ensemble_means,
            ensemble_scales,
        )
        decomposition_length = len(next(iter(decomposition.values())))
        if decomposition_length != len(result):
            raise ValueError(
                "Uncertainty decomposition length mismatch. "
                f"table={len(result)}, decomposition={decomposition_length}"
            )

        if self.residual_base_col is None:
            if 0.5 in quantile_map:
                result[prediction_col] = quantile_map[0.5]
            else:
                result[prediction_col] = pred_quantiles[0]
            _write_quantile_columns(prediction_col, quantile_map)

            if 0.025 in quantile_map:
                result[f"{prediction_col}_lower"] = quantile_map[0.025]
            if 0.975 in quantile_map:
                result[f"{prediction_col}_upper"] = quantile_map[0.975]

            _write_uncertainty_columns(result, prediction_col, decomposition)

            if request_ensemble_mean_columns:
                result[f"{prediction_col}_ensemble_mean"] = mean_arr
            return result

        if self.residual_base_col not in result.columns:
            raise ValueError(
                f"Cannot restore final prediction because `{self.residual_base_col}` is missing."
            )

        base_series = pd.to_numeric(result[self.residual_base_col], errors="coerce")
        base_values = base_series.to_numpy(dtype=np.float32)
        base_valid = np.isfinite(base_values)

        result[f"{prediction_col}_base"] = base_values
        result[f"{prediction_col}_base_missing"] = ~base_valid

        delta_col = f"{prediction_col}_delta"
        if 0.5 in quantile_map:
            result[delta_col] = quantile_map[0.5]
        else:
            result[delta_col] = pred_quantiles[0]
        _write_quantile_columns(delta_col, quantile_map)

        if 0.025 in quantile_map:
            result[f"{delta_col}_lower"] = quantile_map[0.025]
        if 0.975 in quantile_map:
            result[f"{delta_col}_upper"] = quantile_map[0.975]
        _write_uncertainty_columns(result, delta_col, decomposition)

        restored_quantile_map = {
            q: np.where(base_valid, base_values + values, np.nan).astype(np.float32)
            for q, values in quantile_map.items()
        }
        if 0.5 in restored_quantile_map:
            result[prediction_col] = restored_quantile_map[0.5]
        else:
            result[prediction_col] = next(iter(restored_quantile_map.values()))
        _write_quantile_columns(prediction_col, restored_quantile_map)

        if 0.025 in restored_quantile_map:
            result[f"{prediction_col}_lower"] = restored_quantile_map[0.025]
        if 0.975 in restored_quantile_map:
            result[f"{prediction_col}_upper"] = restored_quantile_map[0.975]

        _write_uncertainty_columns(result, prediction_col, decomposition)

        if request_ensemble_mean_columns:
            result[f"{delta_col}_ensemble_mean"] = mean_arr
            result[f"{prediction_col}_ensemble_mean"] = np.where(
                base_valid,
                base_values + mean_arr,
                np.nan,
            ).astype(np.float32)

        return result

    def evaluate_dataframe(
        self,
        table: pd.DataFrame,
        quantiles=(0.025, 0.5, 0.975),
        prediction_col: str = "predict_pm10",
        approximate_quantiles: bool = False,
        prediction_batch_size: int = 2048,
        coverage_levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
        calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Predict a dataframe and evaluate on rows with non-null evaluation targets."""
        result = self.predict_dataframe(
            table,
            quantiles=quantiles,
            prediction_col=prediction_col,
            approximate_quantiles=approximate_quantiles,
            prediction_batch_size=prediction_batch_size,
            keep_ensemble_means=True,
        )

        eval_target_col = (
            self.evaluation_target_col
            if self.evaluation_target_col is not None
            else self.target_col
        )
        if eval_target_col not in result.columns:
            raise ValueError(
                f"Cannot evaluate because `{eval_target_col}` is missing from the dataframe."
            )

        arrays = _extract_evaluation_arrays(result, prediction_col, eval_target_col)
        mask = arrays["mask"]
        if int(mask.sum()) == 0:
            raise ValueError(
                f"No valid rows remain for evaluation after filtering non-null `{eval_target_col}`."
            )

        y_true = arrays["y_true"][mask].astype(np.float64)
        y_pred = arrays["y_pred"][mask].astype(np.float64)
        absolute_error = np.abs(y_true - y_pred)
        squared_error = np.square(y_true - y_pred)

        result["absolute_error"] = np.nan
        result["squared_error"] = np.nan
        result.loc[mask, "absolute_error"] = absolute_error.astype(np.float32)
        result.loc[mask, "squared_error"] = squared_error.astype(np.float32)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
        ss_res = float(np.sum(np.square(y_true - y_pred)))
        ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "smape": _smape_value(y_true, y_pred),
            "n_eval": int(mask.sum()),
            "n_total": int(len(result)),
            "target_col": eval_target_col,
            "model_target_col": self.target_col,
            "prediction_col": prediction_col,
            "residual_base_col": self.residual_base_col,
            "prediction_mode": (
                "residual_restored_absolute"
                if self.residual_base_col is not None
                else "direct"
            ),
        }
        base_missing_col = f"{prediction_col}_base_missing"
        if base_missing_col in result.columns:
            metrics["n_base_missing"] = int(result[base_missing_col].sum())
        metrics.update(
            compute_calibration_metrics(
                result,
                prediction_col=prediction_col,
                target_col=eval_target_col,
                coverage_levels=coverage_levels,
                calibration_bins=calibration_bins,
            )
        )
        metrics.update(_uncertainty_summary_metrics(result, prediction_col))
        return metrics, result

    def fit(self, table, seed):
        raise NotImplementedError("Should be implemented by subclass")

    def likelihood_model(self, table: pd.DataFrame) -> tfd.Distribution:
        if self.params_ is None:
            raise ValueError("The model is not fitted yet. Call `fit` first.")
        test_data = self.data_handler.get_test(table)
        mlp, mlp_template = inference.make_model(**self._model_args(test_data.shape))
        for _ in range(self._ensemble_dims - 1):
            mlp.apply = jax.vmap(mlp.apply, in_axes=(0, None))
        mlp.apply = jax.pmap(mlp.apply, in_axes=(0, None))

        try:
            from . import models
        except ImportError:
            import models

        return models.make_likelihood_model(
            self.params_,
            jnp.asarray(test_data),
            mlp,
            mlp_template,
            self.observation_model,
        )


class BayesianNeuralFieldMAP(BayesianNeuralFieldEstimator):
    """Fits models using stochastic ensembles of maximum-a-posteriori estimates."""

    _ensemble_dims = 2

    def fit(
        self,
        table,
        seed,
        ensemble_size=16,
        learning_rate=0.001,
        num_epochs=5_000,
        batch_size=None,
        num_splits=1,
    ) -> "BayesianNeuralFieldEstimator":
        if ensemble_size < jax.device_count():
            raise ValueError(
                "ensemble_size cannot be smaller than device_count. "
                f"Got ensemble_size={ensemble_size}, device_count={jax.device_count()}."
            )
        train_data = self.data_handler.get_train(table)
        train_target = self.data_handler.get_target(table)

        if len(train_data) != len(train_target):
            raise ValueError(
                f"train_data and train_target have inconsistent lengths: "
                f"{len(train_data)} vs {len(train_target)}"
            )
        if len(train_target) == 0:
            raise ValueError("Training split contains zero labelled samples.")

        if batch_size is None:
            batch_size = train_data.shape[0]
        batch_size = min(int(batch_size), int(train_data.shape[0]))
        batch_size = max(batch_size, 1)

        if self._scale_epochs_by_batch_size:
            num_epochs = num_epochs * max(train_data.shape[0] // batch_size, 1)
        model_args = self._model_args((batch_size, train_data.shape[-1]))
        self.params_, self.losses_ = inference.fit_map(
            train_data,
            train_target,
            seed=seed,
            observation_model=self.observation_model,
            model_args=model_args,
            num_particles=ensemble_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            prior_weight=self._prior_weight,
            batch_size=batch_size,
            num_splits=num_splits,
            variance_regularization_weight=self.variance_regularization_weight,
            variance_reference_scale_factor=self.variance_reference_scale_factor,
        )
        return self
