# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import logging
from collections.abc import Callable, Mapping
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxmod.utils import get_batch_size
from jaxtyping import Array, ArrayLike, Float

from atmodeller.containers import (
    FugacityConstraintSet,
    MassConstraintSet,
    Planet,
    SolverParameters,
    SpeciesCollection,
)
from atmodeller.interfaces import (
    FugacityConstraintProtocol,
    SpeciesProtocol,
    ThermodynamicStateProtocol,
)
from atmodeller.reactions import ReactionSystem

logger: logging.Logger = logging.getLogger(__name__)


class Parameters(eqx.Module):
    """Parameters

    Args:
        reaction_system: Full reaction network
        state: Thermodynamic state
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
        solver_parameters: Solver parameters
        batch_size: Batch size. Defaults to ``1``.
        dilute_limit: Whether to treat dissolution in the dilute limit. Defaults to ``True``.
    """

    reaction_system: ReactionSystem
    """Full reaction network"""
    state: ThermodynamicStateProtocol
    """Thermodynamic state"""
    fugacity_constraints: FugacityConstraintSet
    """Fugacity constraints"""
    mass_constraints: MassConstraintSet
    """Mass constraints"""
    solver_parameters: SolverParameters
    """Solver parameters"""
    batch_size: int = 1
    """Batch size"""

    @classmethod
    def create(
        cls,
        reaction_system: ReactionSystem,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance

        Args:
            reaction_system: Full reaction network
            state: Thermodynamic state. Defaults to a new instance of ``Planet``.
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                a new instance of ``FugacityConstraints``.
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to
                a new instance of ``MassConstraints``.
            solver_parameters: Solver parameters. Defaults to a new instance of
                ``SolverParameters``.

        Returns:
            An instance
        """
        state_: ThermodynamicStateProtocol = Planet() if state is None else state
        fugacity_constraints_: FugacityConstraintSet = FugacityConstraintSet.create(
            reaction_system.species, fugacity_constraints
        )
        mass_constraints_: MassConstraintSet = MassConstraintSet.create(
            reaction_system.species, mass_constraints
        )

        # These pytrees only contain arrays intended for vectorisation (no hidden JAX/NumPy arrays
        # that should remain scalar)
        batch_size: int = get_batch_size((state, fugacity_constraints, mass_constraints))
        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )
        # Always broadcast tau so we can apply vmap to the solver once, even if some calculations
        # need to be repeated due to failures.
        tau_broadcasted: Float[Array, " batch"] = jnp.broadcast_to(
            solver_parameters_.tau, (batch_size,)
        )
        get_leaf: Callable = lambda t: t.tau  # noqa: E731
        solver_parameters_ = eqx.tree_at(get_leaf, solver_parameters_, tau_broadcasted)

        return cls(
            reaction_system,
            state_,
            fugacity_constraints_,
            mass_constraints_,
            solver_parameters_,
            batch_size,
        )

    @property
    def species(self) -> SpeciesCollection[SpeciesProtocol]:
        """Species collection"""
        return self.reaction_system.species
