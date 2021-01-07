# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.navier_stokes_unsteady.navier_stokes_unsteady_problem import NavierStokesUnsteadyProblem
from rbnics.reduction_methods.base import NonlinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.navier_stokes_unsteady.navier_stokes_unsteady_reduction_method import (
    NavierStokesUnsteadyReductionMethod)
from rbnics.reduction_methods.navier_stokes import NavierStokesPODGalerkinReduction
from rbnics.reduction_methods.stokes_unsteady.stokes_unsteady_pod_galerkin_reduction import (
    AbstractCFDUnsteadyPODGalerkinReduction)
from rbnics.utils.decorators import ReductionMethodFor

NavierStokesUnsteadyPODGalerkinReduction_Base = AbstractCFDUnsteadyPODGalerkinReduction(
    NavierStokesPODGalerkinReduction,
    NonlinearTimeDependentPODGalerkinReduction(NavierStokesUnsteadyReductionMethod(NavierStokesPODGalerkinReduction)))


@ReductionMethodFor(NavierStokesUnsteadyProblem, "PODGalerkin")
class NavierStokesUnsteadyPODGalerkinReduction(NavierStokesUnsteadyPODGalerkinReduction_Base):
    pass
