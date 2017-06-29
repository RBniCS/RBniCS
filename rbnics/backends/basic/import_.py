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

# Returns True if it was possible to import the file, False otherwise
def import_(solution, directory, filename, suffix, component, backend, wrapping):
    assert isinstance(solution, (backend.Function.Type(), backend.Matrix.Type(), backend.Vector.Type()))
    if isinstance(solution, backend.Function.Type()):
        if component is None:
            return wrapping.function_load(solution, directory, filename, suffix=suffix)
        else:
            space = wrapping.get_function_space(solution)
            subspace = wrapping.get_function_subspace(solution, component)
            restricted_solution = wrapping.function_extend_or_restrict(solution, component, subspace, None, weight=None, copy=True)
            restricted_solution_loaded = wrapping.function_load(restricted_solution, directory, filename, suffix=suffix)
            wrapping.function_extend_or_restrict(restricted_solution, None, space, component, weight=None, copy=True, extended_or_restricted_function=solution)
            return restricted_solution_loaded
    elif isinstance(solution, (backend.Matrix.Type(), backend.Vector.Type())):
        assert component is None
        assert suffix is None
        return wrapping.tensor_load(solution, directory, filename)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in export.")
