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

from rbnics.eim.reduction_methods.deim_decorated_reduction_method import DEIMDecoratedReductionMethod
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.eim.reduction_methods.eim_decorated_reduction_method import EIMDecoratedReductionMethod
from rbnics.eim.reduction_methods.exact_parametrized_functions_decorated_reduction_method import ExactParametrizedFunctionsDecoratedReductionMethod
from rbnics.eim.reduction_methods.time_dependent_eim_approximation_reduction_method import TimeDependentEIMApproximationReductionMethod

__all__ = [
    'DEIMDecoratedReductionMethod',
    'EIMApproximationReductionMethod',
    'EIMDecoratedReductionMethod',
    'ExactParametrizedFunctionsDecoratedReductionMethod',
    'TimeDependentEIMApproximationReductionMethod'
]
