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
    Planet,
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
        gas_phase: GasPhase,
        melt_phase: Optional[MeltPhase] = None,
        solid_phase: Optional[SolidPhase] = None,
        condensate_phases: Optional[Iterable[PurePhase]] = None,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance from individual phase definitions.

        Args:
            gas_phase: Gas phase
            melt_phase: Melt phase. Defaults to an empty melt phase if not provided.
            solid_phase: Solid phase. Defaults to an empty solid phase if not provided.
            condensate_phases: Pure condensate phases. Defaults to an empty tuple if not provided.
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
        if melt_phase is None:
            melt_phase = MeltPhase.empty()
        if solid_phase is None:
            solid_phase = SolidPhase.empty()
        if condensate_phases is None:
            condensate_phases = ()

        reaction_system: ReactionSystem = ReactionSystem(
            gas_phase=gas_phase,
            melt_phase=melt_phase,
            solid_phase=solid_phase,
            condensate_phases=condensate_phases,
        )

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
        """Species collection"""
        return self.reaction_system.species
