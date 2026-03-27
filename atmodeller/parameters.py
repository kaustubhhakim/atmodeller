# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import logging
from collections.abc import Mapping
from typing import Optional

import equinox as eqx
from jaxtyping import ArrayLike

from atmodeller.containers import (
    FugacityConstraintSet,
    MassConstraintSet,
    SolverParameters,
    SpeciesCollection,
)
from atmodeller.interfaces import FugacityConstraintProtocol, SpeciesProtocol
from atmodeller.jaxhelper import get_batch_size
from atmodeller.reactions import ReactionSystem
from atmodeller.state import ThermodynamicStateProtocol

logger: logging.Logger = logging.getLogger(__name__)


class Parameters(eqx.Module):
    """Parameters

    Args:
        state: Thermodynamic state
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
        solver_parameters: Solver parameters
        batch_size: Batch size. Defaults to ``1``.
    """

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
        state: ThermodynamicStateProtocol,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance from a pre-built reaction system.

        Args:
            state: Thermodynamic state
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                a new instance of ``FugacityConstraints``.
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to
                a new instance of ``MassConstraints``.
            solver_parameters: Solver parameters. Defaults to a new instance of
                ``SolverParameters``.

        Returns:
            An instance
        """
        fugacity_constraints_: FugacityConstraintSet = FugacityConstraintSet.create(
            state.reaction_system.phase_system.species, fugacity_constraints
        )
        mass_constraints_: MassConstraintSet = MassConstraintSet.create(
            state.reaction_system.phase_system.species, mass_constraints
        )
        # TODO: NOTE: fugacity_constraints and not fugacity_constraints_ here. Potentially to
        # change during a refactor.
        batch_size: int = get_batch_size((state, fugacity_constraints, mass_constraints_))
        # jax.debug.print("batch_size (parameters) = {out}", out=batch_size)

        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )

        return cls(state, fugacity_constraints_, mass_constraints_, solver_parameters_, batch_size)

    @property
    def reaction_system(self) -> ReactionSystem:
        """Reaction system representing the thermodynamic state of the planetary body"""
        return self.state.reaction_system

    @property
    def species(self) -> SpeciesCollection[SpeciesProtocol]:
        """Species in the system"""
        return self.reaction_system.phase_system.species

    @property
    def species_names(self) -> tuple[str, ...]:
        """Species names in the system"""
        return self.species.species_names

    @property
    def element_names(self) -> tuple[str, ...]:
        """Unique elements in the system"""
        return self.reaction_system.species.unique_elements

    # TODO: Generalise to update other quantities on the parameters object
    def update_constraints(
        self, new_mass_constraints: Optional[Mapping[str, ArrayLike]] = None
    ) -> "Parameters":
        """Updates the constraints of the parameters.

        Args:
            new_mass_constraints: New mass constraints. Defaults to ``None``.

        Returns:
            Updated parameters.
        """
        parameters_updated: Parameters = self

        if new_mass_constraints is not None:
            mass_constraints_updated: MassConstraintSet = self.mass_constraints.update_abundance(
                new_mass_constraints
            )
            parameters_updated = eqx.tree_at(
                lambda p: p.mass_constraints, self, mass_constraints_updated
            )

        return parameters_updated
