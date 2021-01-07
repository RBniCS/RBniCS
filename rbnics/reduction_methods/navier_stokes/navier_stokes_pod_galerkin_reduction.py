# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.reduction_methods.base import NonlinearPODGalerkinReduction
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.reduction_methods.stokes import StokesPODGalerkinReduction
from rbnics.reduction_methods.navier_stokes.navier_stokes_reduction_method import NavierStokesReductionMethod

NavierStokesPODGalerkinReduction_Base = NonlinearPODGalerkinReduction(
    NavierStokesReductionMethod(StokesPODGalerkinReduction))


@ReductionMethodFor(NavierStokesProblem, "PODGalerkin")
class NavierStokesPODGalerkinReduction(NavierStokesPODGalerkinReduction_Base):
    pass
