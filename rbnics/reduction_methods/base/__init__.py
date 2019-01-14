# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from rbnics.reduction_methods.base.differential_problem_reduction_method import DifferentialProblemReductionMethod
from rbnics.reduction_methods.base.linear_pod_galerkin_reduction import LinearPODGalerkinReduction
from rbnics.reduction_methods.base.linear_rb_reduction import LinearRBReduction
from rbnics.reduction_methods.base.linear_reduction_method import LinearReductionMethod
from rbnics.reduction_methods.base.linear_time_dependent_pod_galerkin_reduction import LinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.base.linear_time_dependent_rb_reduction import LinearTimeDependentRBReduction
from rbnics.reduction_methods.base.linear_time_dependent_reduction_method import LinearTimeDependentReductionMethod
from rbnics.reduction_methods.base.nonlinear_pod_galerkin_reduction import NonlinearPODGalerkinReduction
from rbnics.reduction_methods.base.nonlinear_rb_reduction import NonlinearRBReduction
from rbnics.reduction_methods.base.nonlinear_reduction_method import NonlinearReductionMethod
from rbnics.reduction_methods.base.nonlinear_time_dependent_pod_galerkin_reduction import NonlinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.base.nonlinear_time_dependent_rb_reduction import NonlinearTimeDependentRBReduction
from rbnics.reduction_methods.base.nonlinear_time_dependent_reduction_method import NonlinearTimeDependentReductionMethod
from rbnics.reduction_methods.base.pod_galerkin_reduction import PODGalerkinReduction
from rbnics.reduction_methods.base.rb_reduction import RBReduction
from rbnics.reduction_methods.base.reduction_method import ReductionMethod
from rbnics.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from rbnics.reduction_methods.base.time_dependent_rb_reduction import TimeDependentRBReduction
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod

__all__ = [
    'DifferentialProblemReductionMethod',
    'LinearPODGalerkinReduction',
    'LinearRBReduction',
    'LinearReductionMethod',
    'LinearTimeDependentPODGalerkinReduction',
    'LinearTimeDependentRBReduction',
    'LinearTimeDependentReductionMethod',
    'NonlinearPODGalerkinReduction',
    'NonlinearRBReduction',
    'NonlinearReductionMethod',
    'NonlinearTimeDependentPODGalerkinReduction',
    'NonlinearTimeDependentRBReduction',
    'NonlinearTimeDependentReductionMethod',
    'PODGalerkinReduction',
    'RBReduction',
    'ReductionMethod',
    'TimeDependentPODGalerkinReduction',
    'TimeDependentRBReduction',
    'TimeDependentReductionMethod'
]
