# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_pod_galerkin_reduced_problem import (
    NonlinearParabolicPODGalerkinReducedProblem)
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_problem import NonlinearParabolicProblem
# from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_rb_reduced_problem import (
#    NonlinearParabolicRBReducedProblem)
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_reduced_problem import NonlinearParabolicReducedProblem

__all__ = [
    "NonlinearParabolicPODGalerkinReducedProblem",
    "NonlinearParabolicProblem",
    # "NonlinearParabolicRBReducedProblem",
    "NonlinearParabolicReducedProblem"
]
