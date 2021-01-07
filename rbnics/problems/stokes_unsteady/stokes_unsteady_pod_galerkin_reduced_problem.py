# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.stokes import StokesPODGalerkinReducedProblem
from rbnics.problems.stokes_unsteady.stokes_unsteady_reduced_problem import StokesUnsteadyReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.stokes_unsteady.stokes_unsteady_problem import StokesUnsteadyProblem
from rbnics.problems.base import LinearTimeDependentPODGalerkinReducedProblem
from rbnics.reduction_methods.stokes_unsteady import StokesUnsteadyPODGalerkinReduction


def AbstractCFDUnsteadyPODGalerkinReducedProblem(AbstractCFDUnsteadyPODGalerkinReducedProblem_Base):
    return AbstractCFDUnsteadyPODGalerkinReducedProblem_Base


StokesUnsteadyPODGalerkinReducedProblem_Base = AbstractCFDUnsteadyPODGalerkinReducedProblem(
    LinearTimeDependentPODGalerkinReducedProblem(StokesUnsteadyReducedProblem(StokesPODGalerkinReducedProblem)))


@ReducedProblemFor(StokesUnsteadyProblem, StokesUnsteadyPODGalerkinReduction)
class StokesUnsteadyPODGalerkinReducedProblem(StokesUnsteadyPODGalerkinReducedProblem_Base):
    pass
