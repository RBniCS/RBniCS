# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.navier_stokes.navier_stokes_pod_galerkin_reduced_problem import (
    NavierStokesPODGalerkinReducedProblem)
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
# from rbnics.problems.navier_stokes.navier_stokes_rb_reduced_problem import NavierStokesRBReducedProblem
from rbnics.problems.navier_stokes.navier_stokes_reduced_problem import NavierStokesReducedProblem

__all__ = [
    "NavierStokesPODGalerkinReducedProblem",
    "NavierStokesProblem",
    # "NavierStokesRBReducedProblem",
    "NavierStokesReducedProblem"
]
