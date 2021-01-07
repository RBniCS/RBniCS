# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.navier_stokes import NavierStokesPODGalerkinReducedProblem
from rbnics.problems.navier_stokes_unsteady.navier_stokes_unsteady_problem import NavierStokesUnsteadyProblem
from rbnics.problems.navier_stokes_unsteady.navier_stokes_unsteady_reduced_problem import (
    NavierStokesUnsteadyReducedProblem)
from rbnics.problems.stokes_unsteady.stokes_unsteady_pod_galerkin_reduced_problem import (
    AbstractCFDUnsteadyPODGalerkinReducedProblem)
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearTimeDependentPODGalerkinReducedProblem
from rbnics.reduction_methods.navier_stokes_unsteady import NavierStokesUnsteadyPODGalerkinReduction

NavierStokesUnsteadyPODGalerkinReducedProblem_Base = AbstractCFDUnsteadyPODGalerkinReducedProblem(
    NonlinearTimeDependentPODGalerkinReducedProblem(
        NavierStokesUnsteadyReducedProblem(NavierStokesPODGalerkinReducedProblem)))


@ReducedProblemFor(NavierStokesUnsteadyProblem, NavierStokesUnsteadyPODGalerkinReduction)
class NavierStokesUnsteadyPODGalerkinReducedProblem(NavierStokesUnsteadyPODGalerkinReducedProblem_Base):
    pass
