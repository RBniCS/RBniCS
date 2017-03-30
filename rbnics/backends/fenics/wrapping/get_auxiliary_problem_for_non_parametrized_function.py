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

import hashlib

def get_auxiliary_problem_for_non_parametrized_function(function):
    if function not in get_auxiliary_problem_for_non_parametrized_function._storage:
        # Only a V attribute is required
        class AuxiliaryProblemForNonParametrizedFunction(object):
            def __init__(self, function):
                self.V = function.function_space()
        # Change the name of the (local) class to (almost) uniquely identify the function.
        # Since the unique FEniCS identifier f_** may change between runs, we use as identifiers
        # a combination of the norms, truncated to the first five significant figures.
        norm_1 = round_to_significant_figures(function.vector().norm("l1"), 5)
        norm_2 = round_to_significant_figures(function.vector().norm("l2"), 5)
        norm_inf = round_to_significant_figures(function.vector().norm("linf"), 5)
        AuxiliaryProblemForNonParametrizedFunction.__name__ = (
            "Function_" + hashlib.sha1(
                (norm_1 + norm_2 + norm_inf).encode("utf-8")
            ).hexdigest()
        )
        get_auxiliary_problem_for_non_parametrized_function._storage[function] = AuxiliaryProblemForNonParametrizedFunction(function)
    return get_auxiliary_problem_for_non_parametrized_function._storage[function]
get_auxiliary_problem_for_non_parametrized_function._storage = dict()

def round_to_significant_figures(x, n):
    return "%.*e" % (n-1, x)
    
