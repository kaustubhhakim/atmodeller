# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers for JAX operations"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float

MAX_EXP_INPUT = jnp.log(jnp.finfo(jnp.float64).max)
"""Maximum x for which exp(x) is finite in 64-bit precision to prevent overflow"""
MIN_EXP_INPUT = jnp.log(jnp.finfo(jnp.float64).tiny)
"""Lower bound for stable exp() before underflow to zero"""


def as_j64(x: ArrayLike | tuple) -> Float[Array, "..."]:
    return jnp.asarray(x, dtype=jnp.float64)


def is_jax_array(element: Any) -> bool:  # pragma: no cover
    return isinstance(element, jax.Array)


def safe_exp(x: ArrayLike) -> Array:
    """Computes a numerically stable elementwise exponential with explicit handling of -inf.

    This function extends ``jnp.exp`` with safeguards for common numerical issues:

    - Clips inputs to prevent overflow of ``exp(x)`` for large positive values.
    - Treats ``-inf`` inputs explicitly and returns 0 for those entries
      (i.e., preserves the identity ``exp(-inf) = 0``).
    - Avoids NaNs by replacing invalid values before applying ``exp``.

    Note:
        This function is intended for computations performed in log-space or masked representations
        where ``-inf`` denotes absent or zero-valued contributions. It preserves standard autodiff
        behavior by default (no gradient suppression), unless explicitly modified.

    Args:
        x: Input array-like values

    Returns:
        Array of the same shape as ``x`` containing ``exp(x)`` computed in a numerically stable way
        with special handling of ``-inf``.
    """
    x = jnp.asarray(x)
    is_neg_inf = jnp.isneginf(x)

    # Replace -inf with something safe before clipping
    x_safe = jnp.where(is_neg_inf, 0.0, x)

    # Apply overflow clipping only to finite values
    x_clipped = jnp.clip(x_safe, a_max=MAX_EXP_INPUT)

    y = jnp.exp(x_clipped)

    # Restore semantics: exp(-inf) = 0
    y = jnp.where(is_neg_inf, 0.0, y)

    # Kill gradients through masked entries
    # y = jnp.where(is_neg_inf, jax.lax.stop_gradient(y), y)

    return y


def masked_logsumexp(
    log_x: Float[Array, "... n"], axis: int = -1, keepdims: bool = True
) -> Float[Array, "..."]:
    """Computes a numerically stable log-sum-exp with explicit masking of -inf values.

    This function extends the standard ``logsumexp`` with support for masked inputs, where ``-inf``
    values are treated as absent (i.e., zero contribution in linear space).

    Key properties:

    - Replaces non-finite values (e.g., ``-inf``) with a large negative finite number
      to ensure numerical stability during computation.
    - Computes the log-sum-exp in a stable manner using JAX's implementation.
    - Preserves the semantics that if all values along the reduction axis are masked,
      the result is ``-inf``.
    - Avoids NaNs and remains compatible with automatic differentiation.

    Note:
        This function is intended for use in log-domain computations (e.g., probabilities,
        thermodynamic quantities) where ``-inf`` encodes zero or absent contributions.

    Args:
        log_x: Input array of log-values, where ``-inf`` indicates masked entries
        axis: Axis or axes over which to compute the log-sum-exp
        keepdims: Whether to retain reduced dimensions with length 1

    Returns:
        An array containing the log-sum-exp over the specified axis, with masked inputs properly
        handled and ``-inf`` returned if all entries are masked.
    """
    dtype = log_x.dtype
    neg_large = jnp.finfo(dtype).min

    is_finite: Bool[Array, "..."] = jnp.isfinite(log_x)

    # Replace -inf with large negative number (safe for autodiff)
    safe_log_x: Float[Array, "..."] = jnp.where(is_finite, log_x, neg_large)

    out: Float[Array, "..."] = logsumexp(safe_log_x, axis=axis, keepdims=keepdims)

    # If everything was masked -> return -inf (strict logic)
    all_invalid: Bool[Array, "..."] = ~jnp.any(is_finite, axis=axis, keepdims=keepdims)
    out: Float[Array, "..."] = jnp.where(all_invalid, -jnp.inf, out)

    # Kill gradients if nothing exists
    # out: Float[Array, "..."] = jnp.where(all_invalid, jax.lax.stop_gradient(out), out)

    return out


def to_hashable(x: Callable) -> Callable:  # pragma: no cover
    """Wraps a callable to make it hashable for JAX transformations.

    This wrapper is useful when passing bound methods of Equinox PyTrees (with JAX arrays as
    attributes) to transformations like :func:`jax.jit`, :func:`jax.vmap`, or :func:`lax.scan`. It
    wraps the callable in a lambda to forward all arguments while avoiding JAX trying to trace the
    method itself. See discussion: https://github.com/patrick-kidger/equinox/issues/1011

    Args:
        x: A callable to wrap

    Returns:
        A hashable lambda forwarding all arguments to the original callable.
    """
    return lambda *args, **kwargs: x(*args, **kwargs)
