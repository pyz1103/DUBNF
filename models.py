# Copyright 2024 The bayesnf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model code for Bayesian-neural-field time-series work."""

import enum
from typing import Callable, Tuple

import flax
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
VARIANCE_EPS = 1e-6
LOG_VARIANCE_MIN = -8.0
LOG_VARIANCE_MAX = 6.0
VARIANCE_OUTPUT_INIT_SCALE = 0.05


class LikelihoodDist(enum.Enum):
    NORMAL = "NORMAL"


def make_seasonal_frequencies(
    seasonality_periods: np.ndarray, num_harmonics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique Fourier frequencies for given periods and harmonics."""
    seasonality_periods = np.array(seasonality_periods, dtype=np.float32)
    if np.any((num_harmonics > seasonality_periods / 2)):
        raise ValueError("Harmonic cannot exceed half seasonal period.")
    if seasonality_periods.shape != num_harmonics.shape:
        raise ValueError("Number of seasonal periods and harmonics must be equal.")
    if len(num_harmonics.shape) != 1:
        raise ValueError(
            "Arguments `num_harmonics` and `seasonality_periods` must be rank 1."
        )
    if seasonality_periods.shape[0] == 0:
        return (np.zeros(0), np.zeros(0))
    harmonics = [np.arange(1, h + 1, dtype=np.float32) for h in num_harmonics]
    frequencies = np.concatenate(
        [h / p for (h, p) in zip(harmonics, seasonality_periods)]
    )
    _, idx = np.unique(frequencies, return_index=True)
    idx_sort = np.sort(idx)
    unique_frequencies = frequencies[idx_sort]
    unique_harmonics = np.concatenate(harmonics)[idx_sort]
    return (unique_frequencies, unique_harmonics)


def make_seasonal_features(
    x: jax.typing.ArrayLike,
    seasonality_periods: np.ndarray,
    num_harmonics: np.ndarray,
    rescale: bool = False,
) -> jnp.ndarray:
    """Returns a set of cos and sin features for each seasonality period."""
    x = jnp.reshape(x, (-1, 1))
    frequencies, harmonics = make_seasonal_frequencies(
        seasonality_periods, num_harmonics
    )
    y = 2 * jnp.pi * frequencies * x
    features = jnp.column_stack((jnp.cos(y), jnp.sin(y)))
    denominator = jnp.tile(harmonics, 2)
    return features / denominator if rescale else features


def make_fourier_features(
    x: jax.typing.ArrayLike, max_degree: int, rescale: bool = False
) -> jnp.ndarray:
    """Returns a set of sine and cosine basis functions."""
    x = jnp.reshape(x, (-1, 1))
    degrees = jnp.arange(max_degree)
    y = 2 * jnp.pi * 2**degrees * x
    features = jnp.column_stack((jnp.cos(y), jnp.sin(y)))
    denominator = jnp.tile(degrees + 1, 2)
    return features / denominator if rescale else features


def get_residual_block_names(depth: int) -> list[str]:
    return [f"residual_block_{i}" for i in range(int(depth))]


prior_base_d = tfd.Logistic


def softplus_inverse(value: float) -> float:
    """Numerically stable inverse softplus for positive scalar initializers."""
    value = max(float(value), VARIANCE_EPS)
    return float(np.log(np.expm1(value)))


def prior_model_fn(mlp_template):
    leaves = jax.tree_util.tree_leaves(mlp_template)
    for p in leaves:
        yield prior_base_d(0.0, jnp.ones_like(p))


def _normal_distribution_params(
    network_outputs: jax.Array,
    min_log_variance: float = LOG_VARIANCE_MIN,
    max_log_variance: float = LOG_VARIANCE_MAX,
    variance_eps: float = VARIANCE_EPS,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split dual-head NORMAL outputs into mean, scale, and log-variance."""
    network_outputs = jnp.asarray(network_outputs, dtype=jnp.float32)
    if network_outputs.shape[-1] != 2:
        raise ValueError(
            "NORMAL heteroscedastic likelihood expects a dual-head output with "
            f"shape (..., 2), but received {network_outputs.shape}."
        )
    mu = network_outputs[..., 0]
    log_variance = jnp.clip(
        network_outputs[..., 1],
        min_log_variance,
        max_log_variance,
    )
    variance = jnp.exp(log_variance) + variance_eps
    scale = jnp.sqrt(variance)
    return mu, scale, log_variance


def make_normal_prediction_stats(
    params: jax.typing.ArrayLike,
    x: jax.Array,
    mlp: nn.Module,
    mlp_template: flax.core.frozen_dict.FrozenDict,
) -> dict[str, jax.Array]:
    """Return per-sample heteroscedastic NORMAL parameters."""
    treedef = jax.tree_util.tree_structure(mlp_template)
    mlp_params = jax.tree_util.tree_unflatten(treedef, params)
    network_outputs = mlp.apply(mlp_params, x)
    mu, scale, log_variance = _normal_distribution_params(network_outputs)
    return {
        "mu": mu,
        "scale": scale,
        "log_variance": log_variance,
        "network_outputs": network_outputs,
    }


def make_likelihood_model(
    params: jax.typing.ArrayLike,
    x: jax.Array,
    mlp: nn.Module,
    mlp_template: flax.core.frozen_dict.FrozenDict,
    distribution: str,
) -> tfd.Distribution:
    """Create the likelihood distribution for model predictions."""
    if LikelihoodDist(distribution) != LikelihoodDist.NORMAL:
        raise AssertionError("Unknown likelihood distribution:", distribution)

    stats = make_normal_prediction_stats(params, x, mlp, mlp_template)
    return tfd.Independent(tfd.Normal(stats["mu"], stats["scale"]), 1)


def make_spatial_features(
    x: jax.typing.ArrayLike,
    ref_points: jnp.ndarray,
    k_neighbors: int = 8,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """Build spatial features by soft kNN aggregation over reference points.

    This replaces exact coordinate matching, which fails for mobile trajectories
    whose coordinates are almost always unseen during prediction.
    """
    if ref_points is None or ref_points.shape[0] == 0:
        return jnp.zeros((x.shape[0], 0), dtype=jnp.asarray(x).dtype)

    coords = jnp.asarray(x)[..., 1:3]
    ref_coords = ref_points[:, :2]
    ref_features = ref_points[:, 2:]

    dist2 = jnp.sum(
        jnp.square(coords[:, jnp.newaxis, :] - ref_coords[jnp.newaxis, :, :]),
        axis=-1,
    )
    k = min(k_neighbors, int(ref_coords.shape[0]))
    neg_top_dist2, top_idx = jax.lax.top_k(-dist2, k)
    top_dist2 = -neg_top_dist2
    top_features = ref_features[top_idx]

    scale = jnp.mean(top_dist2, axis=-1, keepdims=True)
    scale = jnp.maximum(scale, eps)
    weights = jax.nn.softmax(-top_dist2 / scale, axis=-1)
    return jnp.sum(top_features * weights[..., jnp.newaxis], axis=1)


class GatedResidualBlock(nn.Module):
    width: int
    activation_fn: Callable[[jax.Array], jax.Array]

    @nn.compact
    def __call__(self, x):
        residual = x
        h = nn.Dense(self.width, kernel_init=nn.initializers.normal(1.0))(x)
        h = self.activation_fn(h)
        h = nn.Dense(self.width, kernel_init=nn.initializers.normal(1.0))(h)
        g = nn.Dense(self.width, kernel_init=nn.initializers.normal(1.0))(x)
        g = nn.sigmoid(g)
        self.sow("intermediates", "gate_values", g)
        return g * h + (1.0 - g) * residual


class ChannelAttention(nn.Module):
    activation_fn: Callable[[jax.Array], jax.Array]
    reduction: int = 8

    @nn.compact
    def __call__(self, x):
        """Apply per-sample channel attention."""
        num_features = x.shape[-1]
        hidden_dim = max(num_features // self.reduction, 1)

        y = nn.Dense(hidden_dim)(x)
        y = self.activation_fn(y)
        y = nn.Dense(num_features)(y)
        y = nn.sigmoid(y)
        self.sow("intermediates", "attention_weights", y)
        return x * y


class BayesianNeuralField1D(nn.Module):
    """Linen Module implementing a 1D Bayesian neural field."""

    width: int
    depth: int
    input_scales: np.ndarray
    fourier_degrees: np.ndarray
    interactions: np.ndarray
    num_seasonal_harmonics: np.ndarray = flax.struct.field(
        default_factory=lambda: np.zeros((0,))
    )
    seasonality_periods: np.ndarray = flax.struct.field(
        default_factory=lambda: np.zeros((0,))
    )
    ref_points: jnp.ndarray | None = None
    spatial_k: int = 8
    use_remat: bool = True
    min_log_variance: float = LOG_VARIANCE_MIN
    max_log_variance: float = LOG_VARIANCE_MAX
    variance_output_init_scale: float = VARIANCE_OUTPUT_INIT_SCALE
    detach_variance_head: bool = True

    @nn.compact
    def __call__(self, x):
        init = nn.initializers.normal(1.0)

        if len(x.shape) == 1:
            x = x[..., jnp.newaxis]

        log_scale_adjustment = self.param("log_scale_adjustment", init, x.shape[-1:])
        scaled_x = x / (self.input_scales * jnp.exp(log_scale_adjustment))

        if self.ref_points is not None and self.ref_points.shape[0] > 0:
            spatial_features = make_spatial_features(
                x, self.ref_points, k_neighbors=self.spatial_k
            )
        else:
            spatial_features = jnp.zeros((x.shape[0], 0), dtype=x.dtype)

        seasonal_features = make_seasonal_features(
            x[..., 0],
            self.seasonality_periods,
            self.num_seasonal_harmonics,
            rescale=True,
        )

        fourier_features = [
            make_fourier_features(scaled_x[..., i], degree, rescale=True)
            for i, degree in enumerate(self.fourier_degrees)
            if degree > 0
        ]

        if self.interactions.shape[0] > 0:
            interaction_features = jnp.prod(scaled_x[:, self.interactions], axis=-1)
        else:
            interaction_features = jnp.zeros((x.shape[0], 0), dtype=x.dtype)

        def make_layer_scale(name, shape=()):
            inv_sp_layer_scale = self.param(name, init, shape)
            return jax.nn.softplus(inv_sp_layer_scale)

        features = [
            scaled_x,
            *fourier_features,
            seasonal_features,
            interaction_features,
            spatial_features,
        ]
        features = [
            f * jax.nn.softplus(self.param(f"feature_inv_sp_scale{i}", init, ()))
            for i, f in enumerate(features)
            if f.size > 0
        ]
        h = jnp.concatenate(features, axis=-1)

        activation_weight = jax.nn.sigmoid(
            self.param("logit_activation_weight", nn.initializers.normal(1.0), ())
        )

        def activation_fn(y):
            return activation_weight * nn.elu(y) + (1.0 - activation_weight) * nn.tanh(
                y
            )

        attention_cls = (
            nn.remat(ChannelAttention) if self.use_remat else ChannelAttention
        )
        residual_cls = (
            nn.remat(GatedResidualBlock) if self.use_remat else GatedResidualBlock
        )

        h = attention_cls(
            activation_fn=activation_fn,
            name="channel_attention_input",
        )(h)
        h = nn.Dense(
            self.width,
            kernel_init=nn.initializers.normal(1.0),
            name="input_projection",
        )(h)

        for block_idx in range(self.depth):
            h = residual_cls(
                width=self.width,
                activation_fn=activation_fn,
                name=f"residual_block_{block_idx}",
            )(h)
            h = attention_cls(
                activation_fn=activation_fn,
                name=f"channel_attention_block_{block_idx}",
            )(h)

        mean_head = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(1.0),
            bias_init=nn.initializers.normal(1.0),
            name="mean_head",
        )
        variance_head = nn.Dense(
            1,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="variance_head",
        )
        mean_output_scale = make_layer_scale("inv_sp_output_scale_mean")
        variance_output_scale = jax.nn.softplus(
            self.param(
                "inv_sp_output_scale_var",
                nn.initializers.constant(
                    softplus_inverse(self.variance_output_init_scale)
                ),
                (),
            )
        )
        base_log_variance = self.param(
            "base_log_variance",
            nn.initializers.constant(0.0),
            (),
        )
        h = h / jnp.sqrt(h.shape[-1])
        mu = mean_output_scale * mean_head(h)[..., 0]
        variance_input = jax.lax.stop_gradient(h) if self.detach_variance_head else h
        log_var = (
            base_log_variance
            + variance_output_scale * variance_head(variance_input)[..., 0]
        )
        log_var = jnp.clip(
            log_var,
            self.min_log_variance,
            self.max_log_variance,
        )
        return jnp.stack([mu, log_var], axis=-1)
