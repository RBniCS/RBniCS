# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.differential_problem_reduction_method import DifferentialProblemReductionMethod
from rbnics.reduction_methods.base.linear_pod_galerkin_reduction import LinearPODGalerkinReduction
from rbnics.reduction_methods.base.linear_rb_reduction import LinearRBReduction
from rbnics.reduction_methods.base.linear_reduction_method import LinearReductionMethod
from rbnics.reduction_methods.base.linear_time_dependent_pod_galerkin_reduction import (
    LinearTimeDependentPODGalerkinReduction)
from rbnics.reduction_methods.base.linear_time_dependent_rb_reduction import LinearTimeDependentRBReduction
from rbnics.reduction_methods.base.linear_time_dependent_reduction_method import LinearTimeDependentReductionMethod
from rbnics.reduction_methods.base.nonlinear_pod_galerkin_reduction import NonlinearPODGalerkinReduction
from rbnics.reduction_methods.base.nonlinear_rb_reduction import NonlinearRBReduction
from rbnics.reduction_methods.base.nonlinear_reduction_method import NonlinearReductionMethod
from rbnics.reduction_methods.base.nonlinear_time_dependent_pod_galerkin_reduction import (
    NonlinearTimeDependentPODGalerkinReduction)
from rbnics.reduction_methods.base.nonlinear_time_dependent_rb_reduction import NonlinearTimeDependentRBReduction
from rbnics.reduction_methods.base.nonlinear_time_dependent_reduction_method import (
    NonlinearTimeDependentReductionMethod)
from rbnics.reduction_methods.base.pod_galerkin_reduction import PODGalerkinReduction
from rbnics.reduction_methods.base.rb_reduction import RBReduction
from rbnics.reduction_methods.base.reduction_method import ReductionMethod
from rbnics.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from rbnics.reduction_methods.base.time_dependent_rb_reduction import TimeDependentRBReduction
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod

__all__ = [
    "DifferentialProblemReductionMethod",
    "LinearPODGalerkinReduction",
    "LinearRBReduction",
    "LinearReductionMethod",
    "LinearTimeDependentPODGalerkinReduction",
    "LinearTimeDependentRBReduction",
    "LinearTimeDependentReductionMethod",
    "NonlinearPODGalerkinReduction",
    "NonlinearRBReduction",
    "NonlinearReductionMethod",
    "NonlinearTimeDependentPODGalerkinReduction",
    "NonlinearTimeDependentRBReduction",
    "NonlinearTimeDependentReductionMethod",
    "PODGalerkinReduction",
    "RBReduction",
    "ReductionMethod",
    "TimeDependentPODGalerkinReduction",
    "TimeDependentRBReduction",
    "TimeDependentReductionMethod"
]
