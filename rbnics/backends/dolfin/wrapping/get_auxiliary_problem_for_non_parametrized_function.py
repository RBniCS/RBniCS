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
from dolfin import Function
import rbnics.backends.dolfin
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component import _remove_mute_indices

def get_auxiliary_problem_for_non_parametrized_function(function, backend=None):
    if backend is None:
        backend = rbnics.backends.dolfin
    
    function = _remove_mute_indices(function)
    
    assert (
        (function in get_auxiliary_problem_for_non_parametrized_function._storage_problem)
            ==
        (function in get_auxiliary_problem_for_non_parametrized_function._storage_component)
    )
    if function not in get_auxiliary_problem_for_non_parametrized_function._storage_problem:
        assert isinstance(function, Function), "The case of split(non parametrized function) has not been implemented yet"
        # Only a V attribute is required
        class AuxiliaryProblemForNonParametrizedFunction(object):
            def __init__(self, function):
                self.V = backend.wrapping.get_function_space(function)
        # Change the name of the (local) class to (almost) uniquely identify the function.
        # Since the unique dolfin identifier f_** may change between runs, we use as identifiers
        # a combination of the norms, truncated to the first five significant figures.
        norm_1 = round_to_significant_figures(backend.wrapping.get_function_norm(function, "l1"), 5)
        norm_2 = round_to_significant_figures(backend.wrapping.get_function_norm(function, "l2"), 5)
        norm_inf = round_to_significant_figures(backend.wrapping.get_function_norm(function, "linf"), 5)
        AuxiliaryProblemForNonParametrizedFunction.__name__ = (
            "Function_" + hashlib.sha1(
                (norm_1 + norm_2 + norm_inf).encode("utf-8")
            ).hexdigest()
        )
        get_auxiliary_problem_for_non_parametrized_function._storage_problem[function] = AuxiliaryProblemForNonParametrizedFunction(function)
        get_auxiliary_problem_for_non_parametrized_function._storage_component[function] = (None, )
    return (
        get_auxiliary_problem_for_non_parametrized_function._storage_problem[function],
        get_auxiliary_problem_for_non_parametrized_function._storage_component[function]
    )
get_auxiliary_problem_for_non_parametrized_function._storage_problem = dict()
get_auxiliary_problem_for_non_parametrized_function._storage_component = dict()

def round_to_significant_figures(x, n):
    return "%.*e" % (n-1, x)
    
