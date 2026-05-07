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

"""Inference utilities for spatiotemporal neural-field ensembles."""

import functools
import inspect
from typing import Any, Callable

import flax
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jaxtyping import PyTree
from tensorflow_probability.substrates import jax as tfp

try:
    from . import models
except ImportError:
    import models


tfd = tfp.distributions
ArrayT = jax.Array | np.ndarray


def _tree_path_to_name(path) -> str:
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return "/".join(parts)


def permute_dataset(
    features: ArrayT, target: ArrayT, seed: jax.Array
) -> tuple[ArrayT, ArrayT]:
    permutation = jax.random.permutation(seed, jnp.arange(target.shape[0]))
    return features[permutation], target[permutation]


@functools.partial(jax.jit, static_argnames=("axis",))
def _normal_quantile_via_root(means, scales, q, axis=(0, 1)):
    n = tfd.Normal(means, scales)
    res = tfp.math.find_root_chandrupatla(
        lambda x: n.cdf(x).mean(axis) - q,
        low=jnp.amin(means) - 5 * jnp.amax(scales),
        high=jnp.amax(means) + 5 * jnp.amax(scales),
        value_tolerance=1e-5,
        max_iterations=60,
    )
    return res.estimated_root


@functools.partial(jax.jit, static_argnames=("axis",))
def _approximate_normal_quantile(
    means: jax.Array, scales: jax.Array, q: float, axis=(0, 1)
) -> jax.Array:
    """Fast approximate quantile for a mixture of gaussians."""
    mixture_mean = means.mean(axis)
    mixture_scale = jnp.sqrt(
        (jnp.square(scales) + jnp.square(means)).mean(axis) - jnp.square(mixture_mean)
    )
    n = tfd.Normal(mixture_mean, mixture_scale)
    return n.quantile(q)


def _get_percentile_normal(
    means, scales, quantiles, axis=(0, 1), approximate=False
) -> list[jax.Array]:
    if approximate:
        quantile_fn = _approximate_normal_quantile
    else:
        quantile_fn = _normal_quantile_via_root

    scales_for_quantile = _align_scales_to_means(means, scales)

    forecast_quantiles = []
    for q in quantiles:
        forecast_quantiles.append(quantile_fn(means, scales_for_quantile, q, axis))
    return forecast_quantiles


def _make_forecast_inner(mlp, mlp_template, distribution):
    """Construct inner forecast function for NORMAL MAP inference."""

    def forecast_inner(params, x_subset):
        likelihood = models.make_likelihood_model(
            params, x_subset, mlp, mlp_template, distribution
        )
        if distribution != models.LikelihoodDist.NORMAL:
            raise TypeError("Distribution must be NORMAL.")
        return (likelihood.distribution.loc, likelihood.distribution.scale)

    return forecast_inner


def _align_scales_to_means(means: jax.Array, scales: jax.Array) -> jax.Array:
    """Broadcast per-member scales to the same shape as ensemble means."""
    means = jnp.asarray(means)
    scales = jnp.asarray(scales, dtype=means.dtype)
    while scales.ndim < means.ndim:
        scales = scales[..., jnp.newaxis]
    if scales.shape == means.shape:
        return scales
    try:
        return jnp.broadcast_to(scales, means.shape)
    except ValueError as exc:
        raise ValueError(
            f"Cannot align scales shape {scales.shape} with means shape {means.shape}."
        ) from exc


def forecast_parameters_batched(
    features: jax.Array,
    params: PyTree,
    distribution: models.LikelihoodDist,
    forecast_inner: Callable[[PyTree, jax.Array], PyTree],
    batchsize: int = 1024,
) -> tuple[jax.Array, ...]:
    """Computes parameters of the likelihood distribution in batches."""
    forecast_params_slices = [[], []]

    data_size = int(features.shape[0])
    if data_size == 0:
        raise ValueError("features must contain at least one row.")

    batchsize = max(1, min(int(batchsize), data_size))
    for start in range(0, data_size, batchsize):
        end = min(start + batchsize, data_size)
        forecast_params = forecast_inner(params, features[start:end])
        if distribution == models.LikelihoodDist.NORMAL:
            forecast_params = (
                forecast_params[0],
                _align_scales_to_means(forecast_params[0], forecast_params[1]),
            )
        for i, fc_param in enumerate(forecast_params):
            forecast_params_slices[i].append(fc_param)

    if distribution != models.LikelihoodDist.NORMAL:
        raise TypeError("Distribution must be NORMAL.")

    loc = jnp.concatenate(forecast_params_slices[0], axis=-1)
    scale = jnp.concatenate(forecast_params_slices[1], axis=-1)
    forecast_params = (loc, scale)

    return tuple(forecast_params)


def make_model(
    width: int,
    depth: int,
    input_scales: np.ndarray,
    num_seasonal_harmonics: np.ndarray,
    seasonality_periods: np.ndarray,
    init_x: tuple[int, ...],
    fourier_degrees: np.ndarray,
    interactions: np.ndarray,
    ref_points: jnp.ndarray | None = None,
    spatial_k: int = 8,
    use_remat: bool = True,
    min_log_variance: float = models.LOG_VARIANCE_MIN,
    max_log_variance: float = models.LOG_VARIANCE_MAX,
    variance_output_init_scale: float = models.VARIANCE_OUTPUT_INIT_SCALE,
    detach_variance_head: bool = True,
) -> tuple[models.BayesianNeuralField1D, flax.core.scope.FrozenVariableDict]:
    """Instantiate and initialize BayesianNeuralField1D model.

    Important: `init_x` only needs feature dimension. Parameter shapes do not
    depend on batch size, so callers should pass `(1, num_features)` to avoid a
    huge dummy forward during `mlp.init()`.
    """
    mlp = models.BayesianNeuralField1D(
        width=width,
        depth=depth,
        input_scales=input_scales,
        fourier_degrees=fourier_degrees,
        interactions=interactions,
        num_seasonal_harmonics=num_seasonal_harmonics,
        seasonality_periods=seasonality_periods,
        ref_points=(ref_points.astype(jnp.float32) if ref_points is not None else None),
        spatial_k=spatial_k,
        use_remat=use_remat,
        min_log_variance=min_log_variance,
        max_log_variance=max_log_variance,
        variance_output_init_scale=variance_output_init_scale,
        detach_variance_head=detach_variance_head,
    )
    mlp_template = mlp.init(jax.random.PRNGKey(0), jnp.zeros(init_x, dtype=jnp.float32))
    return mlp, mlp_template


def make_prior(
    mlp_template: flax.core.scope.FrozenVariableDict | None = None,
    **kwargs: dict[str, Any],
) -> tfd.JointDistributionCoroutine:
    if mlp_template is None:
        model_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.getfullargspec(make_model).args
        }
        _, mlp_template = make_model(**model_kwargs)
    prior_d = tfd.JointDistributionCoroutine(
        functools.partial(models.prior_model_fn, mlp_template),
        use_vectorized_map=True,
        batch_ndims=0,
    )
    return prior_d


def fit_map(
    features: ArrayT,
    target: ArrayT,
    seed: jax.Array,
    observation_model: str,
    model_args: dict[str, Any],
    num_particles: int,
    learning_rate: float,
    num_epochs: int,
    prior_weight: float = 1.0,
    batch_size: int | None = None,
    num_splits: int = 1,
    variance_regularization_weight: float = 1e-4,
    variance_reference_scale_factor: float = 0.5,
) -> tuple[PyTree, np.ndarray]:
    """Fit a MAP ensemble using feature and target arrays."""
    distribution = models.LikelihoodDist(observation_model)
    mlp, mlp_template = make_model(**model_args)
    target_scale = max(
        float(np.std(np.asarray(target, dtype=np.float32))),
        float(np.sqrt(models.VARIANCE_EPS)),
    )
    variance_reference_std = max(
        target_scale * float(variance_reference_scale_factor),
        float(np.sqrt(models.VARIANCE_EPS)),
    )
    reference_log_variance = float(np.log(np.square(variance_reference_std)))

    def _neg_energy_fn(params, x, y):
        if distribution != models.LikelihoodDist.NORMAL:
            raise TypeError("Distribution must be NORMAL.")
        stats = models.make_normal_prediction_stats(params, x, mlp, mlp_template)
        log_prob = tfd.Independent(
            tfd.Normal(stats["mu"], stats["scale"]),
            1,
        ).log_prob(y)
        if variance_regularization_weight > 0.0:

            variance_penalty = jnp.mean(
                jnp.square(stats["log_variance"] - reference_log_variance)
            )
            log_prob = log_prob - (
                variance_regularization_weight * x.shape[0] * variance_penalty
            )
        return log_prob

    def _make_init_fn(prior_d):
        xs = prior_d.sample(seed=jax.random.PRNGKey(0))
        leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(mlp_template)
        leaf_names = [_tree_path_to_name(path) for path, _ in leaves_with_path]
        if len(xs) != len(leaf_names):
            raise ValueError(
                "Prior sample and model template disagree on leaf count: "
                f"{len(xs)} vs {len(leaf_names)}."
            )
        variance_output_scale_init = jnp.asarray(
            models.softplus_inverse(
                float(
                    model_args.get(
                        "variance_output_init_scale",
                        models.VARIANCE_OUTPUT_INIT_SCALE,
                    )
                )
            ),
            dtype=jnp.float32,
        )
        base_log_variance_init = jnp.asarray(
            reference_log_variance,
            dtype=jnp.float32,
        )

        def _fn():
            for i, (x, leaf_name) in enumerate(zip(xs, leaf_names)):
                if leaf_name.endswith("base_log_variance"):
                    yield tfd.Deterministic(
                        jnp.ones_like(x) * base_log_variance_init,
                        name="initial_base_log_variance",
                    )
                elif leaf_name.endswith("inv_sp_output_scale_var"):
                    yield tfd.Deterministic(
                        jnp.ones_like(x) * variance_output_scale_init,
                        name="initial_variance_output_scale",
                    )
                elif leaf_name.endswith("variance_head/kernel") or leaf_name.endswith(
                    "variance_head/bias"
                ):
                    yield tfd.Deterministic(
                        jnp.zeros_like(x),
                        name=f"stable_initial_variance_head_{i}",
                    )
                elif len(x.shape) != 2:
                    yield tfd.Deterministic(
                        jnp.zeros_like(x),
                        name=f"zero_initial_mean_for_bias_or_transformed_scale_{i}",
                    )
                else:
                    yield tfd.TruncatedNormal(
                        0.0,
                        jnp.ones_like(x),
                        low=-2,
                        high=2,
                        name=f"initial_weight_matrix_{i}",
                    )

        return lambda seed: tfd.JointDistributionCoroutine(
            _fn, use_vectorized_map=True, batch_ndims=0
        ).sample(seed=seed)

    prior = make_prior(mlp_template=mlp_template)
    params = []
    losses = []
    for i in range(num_splits):
        seed_i = jax.random.fold_in(seed, i) if num_splits > 1 else seed
        params_i, losses_i = ensemble_map(
            features,
            target,
            _neg_energy_fn,
            prior_d=prior,
            init_fn=_make_init_fn(prior),
            ensemble_size=((num_particles // num_splits) // jax.device_count()),
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            seed=seed_i,
            batch_size=batch_size,
            prior_weight=prior_weight,
        )
        params.append(jax.tree_util.tree_map(np.asarray, params_i))
        losses.append(np.asarray(losses_i))

    params = jax.tree_util.tree_map(lambda *ts: np.concatenate(ts, axis=1), *params)
    losses = np.concatenate(losses, axis=1)
    return params, losses


def _make_parallel_forecast_inner(
    mlp,
    mlp_template,
    distribution: models.LikelihoodDist,
    ensemble_dims: int,
):
    forecast_inner = _make_forecast_inner(mlp, mlp_template, distribution)
    for _ in range(ensemble_dims - 1):
        forecast_inner = jax.vmap(forecast_inner, in_axes=(0, None))
    forecast_inner = jax.pmap(forecast_inner, in_axes=(0, None))
    return forecast_inner


def predict_bnf(
    features: ArrayT,
    observation_model: str,
    params: PyTree,
    model_args: dict[str, Any],
    quantiles: jax.Array,
    ensemble_dims: int = 2,
    approximate_quantiles: bool = False,
    batch_size: int = 2048,
    keep_ensemble_means: bool = True,
    keep_ensemble_scales: bool = False,
    return_aux: bool = False,
) -> (
    tuple[jax.Array | None, list[jax.Array]]
    | tuple[jax.Array | None, list[jax.Array], dict[str, jax.Array | None]]
):
    """Predict new data from a fitted ensemble."""
    distribution = models.LikelihoodDist(observation_model)
    assert ensemble_dims >= 1

    features = jnp.asarray(features, dtype=jnp.float32)
    mlp, mlp_template = make_model(**model_args)
    forecast_inner = _make_parallel_forecast_inner(
        mlp, mlp_template, distribution, ensemble_dims
    )
    axis = tuple(range(ensemble_dims))

    data_size = int(features.shape[0])
    if data_size == 0:
        empty_quantiles = [np.zeros((0,), dtype=np.float32) for _ in quantiles]
        if return_aux:
            return (
                None,
                empty_quantiles,
                {
                    "ensemble_means": None,
                    "ensemble_scales": None,
                },
            )
        return None, empty_quantiles
    batch_size = max(1, min(int(batch_size), data_size))

    mean_slices = []
    scale_slices = []
    quantile_slices = [[] for _ in quantiles]

    for start in range(0, data_size, batch_size):
        end = min(start + batch_size, data_size)
        batch_x = features[start:end]
        forecast_params = forecast_inner(params, batch_x)

        if distribution != models.LikelihoodDist.NORMAL:
            raise ValueError(f"Unknown distribution: {distribution}")

        means, scales = forecast_params
        scales = _align_scales_to_means(means, scales)
        batch_quantiles = _get_percentile_normal(
            means,
            scales,
            quantiles,
            axis=axis,
            approximate=approximate_quantiles,
        )
        if keep_ensemble_means:
            mean_slices.append(np.asarray(means))
        if keep_ensemble_scales:
            scale_slices.append(np.asarray(scales))

        for i, q_arr in enumerate(batch_quantiles):
            quantile_slices[i].append(np.asarray(q_arr))

    if keep_ensemble_means:
        forecast_means = np.concatenate(mean_slices, axis=-1)
    else:
        forecast_means = None
    forecast_quantiles = [np.concatenate(v, axis=-1) for v in quantile_slices]
    if keep_ensemble_scales:
        forecast_scales = np.concatenate(scale_slices, axis=-1)
    else:
        forecast_scales = None
    if return_aux:
        return (
            forecast_means,
            forecast_quantiles,
            {
                "ensemble_means": forecast_means,
                "ensemble_scales": forecast_scales,
            },
        )
    return forecast_means, forecast_quantiles


def ensemble_map(
    features: ArrayT,
    target: ArrayT,
    neg_energy_fn: Callable[[PyTree, jax.Array, jax.Array], float],
    prior_d: tfd.Distribution,
    init_fn: Callable[[jax.Array], PyTree],
    ensemble_size: int,
    learning_rate: float,
    num_epochs: int,
    seed: jax.Array,
    batch_size: int | None = None,
    prior_weight: float = 1.0,
) -> tuple[PyTree, jax.Array]:
    """Fit an ensemble of MAP estimates."""
    features = jnp.asarray(features, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    if batch_size is None:
        batch_size = target.shape[0]

    def _target_log_prob_fn(params, x_batch, y_batch):
        if prior_weight == 0.0:
            return -(
                neg_energy_fn(params, x_batch, y_batch) * (target.shape[0] / batch_size)
            )
        else:
            return -(
                neg_energy_fn(params, x_batch, y_batch) * (target.shape[0] / batch_size)
                + prior_d.log_prob(params) * prior_weight
            )

    init_seed, opt_seed = jax.random.split(seed, 2)
    num_devices = jax.device_count()
    init_params = jax.vmap(jax.vmap(init_fn))(
        jax.random.split(init_seed, (num_devices, ensemble_size))
    )

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(num_epochs * max(target.shape[0] // batch_size, 1), 1),
        alpha=0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=1e-4),
    )

    @jax.pmap
    @jax.vmap
    def _run(init_params, seed):
        opt_state = optimizer.init(init_params)

        def _reshape_to_batches(t):
            t = jax.tree_util.tree_map(
                lambda v: v[: (v.shape[0] // batch_size) * batch_size], t
            )
            return jax.tree_util.tree_map(
                lambda v: v.reshape((-1, batch_size) + v.shape[1:]), t
            )

        def _one_epoch(carry, _):
            params, opt_state, seed = carry
            seed, permute_seed = jax.random.split(seed, 2)
            if batch_size < target.shape[0]:
                x, y = permute_dataset(features, target, permute_seed)
            else:
                x, y = features, target

            def _one_step(carry, batch):
                params, opt_state = carry
                batch_x, batch_y = batch
                loss, grads = jax.value_and_grad(_target_log_prob_fn)(
                    params, batch_x, batch_y
                )
                updates, opt_state = optimizer.update(grads, opt_state, params=params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss

            (params, opt_state), losses = jax.lax.scan(
                _one_step,
                (params, opt_state),
                (_reshape_to_batches(x), _reshape_to_batches(y)),
            )
            return (params, opt_state, seed), losses.mean()

        (params, _, _), losses = jax.lax.scan(
            _one_epoch, (init_params, opt_state, seed), None, length=num_epochs
        )
        return params, losses

    return _run(init_params, jax.random.split(opt_seed, (num_devices, ensemble_size)))
