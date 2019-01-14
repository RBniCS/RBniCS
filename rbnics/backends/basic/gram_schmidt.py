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

from math import sqrt
from rbnics.backends.abstract import GramSchmidt as AbstractGramSchmidt
from rbnics.utils.decorators import dict_of, overload

def GramSchmidt(backend, wrapping):
    class _GramSchmidt(AbstractGramSchmidt):
        def __init__(self, space, inner_product, component=None):
            if component is None:
                self.space = space
            else:
                self.space = wrapping.get_function_subspace(space, component)
            self.inner_product = inner_product
            
        def apply(self, new_basis_function, basis_functions, component=None):
            inner_product = self.inner_product
            
            transpose = backend.transpose
            
            new_basis_function = self._extend_or_restrict_if_needed(new_basis_function, component)

            for b in basis_functions:
                new_basis_function = wrapping.gram_schmidt_projection_step(new_basis_function, inner_product, b, transpose)
            norm_new_basis_function = sqrt(transpose(new_basis_function)*inner_product*new_basis_function)
            if norm_new_basis_function != 0.:
                new_basis_function /= norm_new_basis_function
                
            return new_basis_function
            
        @overload(backend.Function.Type(), (None, str))
        def _extend_or_restrict_if_needed(self, function, component):
            return wrapping.function_extend_or_restrict(function, component, self.space, component, weight=None, copy=True)
            
        @overload(backend.Function.Type(), dict_of(str, str))
        def _extend_or_restrict_if_needed(self, function, component):
            assert len(component) == 1
            for (component_from, component_to) in component.items():
                break
            return wrapping.function_extend_or_restrict(function, component_from, self.space, component_to, weight=None, copy=True)
    return _GramSchmidt
