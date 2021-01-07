# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.stokes import StokesPODGalerkinReducedProblem
from rbnics.problems.navier_stokes.navier_stokes_reduced_problem import NavierStokesReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearPODGalerkinReducedProblem
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.reduction_methods.navier_stokes import NavierStokesPODGalerkinReduction

NavierStokesPODGalerkinReducedProblem_Base = NonlinearPODGalerkinReducedProblem(
    NavierStokesReducedProblem(StokesPODGalerkinReducedProblem))


@ReducedProblemFor(NavierStokesProblem, NavierStokesPODGalerkinReduction)
class NavierStokesPODGalerkinReducedProblem(NavierStokesPODGalerkinReducedProblem_Base):
    pass
