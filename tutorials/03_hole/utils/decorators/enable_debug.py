# Copyright (C) 2015-2020 by the RBniCS authors
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

import sys
from logging import basicConfig, DEBUG
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import (
    logger as pull_back_to_reference_domain_logger)
from rbnics.shape_parametrization.problems.affine_shape_parametrization_decorated_problem import (
    logger as affine_shape_parametrization_logger)


def EnableDebug():
    basicConfig(stream=sys.stdout, format="%(message)s")
    affine_shape_parametrization_logger.setLevel(DEBUG)
    pull_back_to_reference_domain_logger.setLevel(DEBUG)

    def EnableDebug_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        # return value (a class) for the decorator
        return ParametrizedDifferentialProblem_DerivedClass

    # return the decorator itself
    return EnableDebug_Decorator
