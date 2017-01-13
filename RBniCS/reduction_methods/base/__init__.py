# Copyright (C) 2015-2017 by the RBniCS authors
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
## @file __init__.py
#  @brief Init file for auxiliary reduction methods module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.reduction_methods.base.differential_problem_reduction_method import DifferentialProblemReductionMethod
from RBniCS.reduction_methods.base.pod_galerkin_reduction import PODGalerkinReduction
from RBniCS.reduction_methods.base.rb_reduction import RBReduction
from RBniCS.reduction_methods.base.reduction_method import ReductionMethod
from RBniCS.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from RBniCS.reduction_methods.base.time_dependent_rb_reduction import TimeDependentRBReduction
from RBniCS.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod

__all__ = [
    'DifferentialProblemReductionMethod',
    'PODGalerkinReduction',
    'RBReduction',
    'ReductionMethod',
    'TimeDependentPODGalerkinReduction',
    'TimeDependentRBReduction',
    'TimeDependentReductionMethod'
]
