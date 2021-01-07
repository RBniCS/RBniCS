# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.stokes.stokes_reduced_problem import StokesReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.stokes.stokes_problem import StokesProblem
from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.stokes import StokesPODGalerkinReduction

StokesPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    StokesReducedProblem(ParametrizedReducedDifferentialProblem))


@ReducedProblemFor(StokesProblem, StokesPODGalerkinReduction)
class StokesPODGalerkinReducedProblem(StokesPODGalerkinReducedProblem_Base):
    pass
