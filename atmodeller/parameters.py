# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameters"""

import logging
from collections.abc import Iterable, Mapping
from typing import Optional

import equinox as eqx
from jaxmod.utils import get_batch_size
from jaxtyping import ArrayLike

from atmodeller.containers import (
    FugacityConstraintSet,
    MassConstraintSet,
    SolverParameters,
    SpeciesCollection,
)
from atmodeller.interfaces import (
    FugacityConstraintProtocol,
    SpeciesProtocol,
    ThermodynamicStateProtocol,
)
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.reactions import ReactionSystem
from atmodeller.state import PhaseSystem, Planet

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
    def from_phases(
        cls,
        gas: GasPhase,
        melt: Optional[MeltPhase] = None,
        solid: Optional[SolidPhase] = None,
        condensates: Optional[Iterable[PurePhase]] = None,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance from individual phase definitions.

        Args:
            gas: Gas phase
            melt: Melt phase. Defaults to an empty melt phase if not provided.
            solid: Solid phase. Defaults to an empty solid phase if not provided.
            condensates: Pure condensate phases. Defaults to an empty tuple if not provided.
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
        phase_system: PhaseSystem = PhaseSystem(
            gas=gas, melt=melt, solid=solid, condensates=condensates
        )
        reaction_system: ReactionSystem = ReactionSystem(phase_system)

        return cls.from_reaction_system(
            reaction_system, state, fugacity_constraints, mass_constraints, solver_parameters
        )

    @classmethod
    def from_reaction_system(
        cls,
        reaction_system: ReactionSystem,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance from a pre-built reaction system.

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
        """Species in the system"""
        return self.reaction_system.species

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
