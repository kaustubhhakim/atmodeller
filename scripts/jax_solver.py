#!/usr/env/bin python
"""Synthetic example using vmap"""
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


# Define a custom pytree class to hold our parameters
@register_pytree_node_class
class LinearSystemParams:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def tree_flatten(self):
        children = (self.A, self.b)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class AdditionalParams:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def tree_flatten(self):
        children = (self.scale_factor,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def solve_linear_system(A, b, additional_params):
    x = jnp.linalg.solve(A, b)
    x_scaled = x * additional_params.scale_factor
    return x_scaled


def solve_batch(params_list, additional_params):
    # Extract the arrays from the list of params and stack them
    A_batch = jnp.stack([params.A for params in params_list])
    b_batch = jnp.stack([params.b for params in params_list])

    # JIT compile the solve function
    jit_solve = jax.jit(solve_linear_system)

    # Vectorize the JIT-compiled solve function
    # Use `in_axes=(0, 0, None)` to indicate that the first two arguments are batched,
    # while the third argument is fixed (broadcasted).
    vmap_solve = jax.vmap(jit_solve, in_axes=(0, 0, None))

    # Apply the vectorized function to the batch of params
    solutions = vmap_solve(A_batch, b_batch, additional_params)
    return solutions


def main():
    # Define a batch of systems of equations
    A_list = jnp.array([[[3, 2], [1, 2]], [[4, 1], [2, 2]], [[2, 3], [1, 1]]])
    b_list = jnp.array([[1, 4], [3, 4], [5, 2]])

    # Define the additional fixed parameters
    additional_params = AdditionalParams(scale_factor=2.0)

    # Create a list of pytree params
    params_list = [LinearSystemParams(A, b) for A, b in zip(A_list, b_list)]

    # Solve the batch of linear systems
    solutions = solve_batch(params_list, additional_params)

    # Print the solutions
    print("Solutions x:", solutions)


if __name__ == "__main__":
    main()
