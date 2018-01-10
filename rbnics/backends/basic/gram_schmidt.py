# Copyright (C) 2015-2018 by the RBniCS authors
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

def GramSchmidt(backend, wrapping):
    class _GramSchmidt(AbstractGramSchmidt):
        def __init__(self, inner_product):
            # Inner product
            self.inner_product = inner_product
            
        def apply(self, basis_functions, N_bc):
            inner_product = self.inner_product
            
            transpose = backend.transpose

            n_basis = len(basis_functions)
            b = basis_functions[n_basis - 1]
            for i in range(N_bc, n_basis - 1):
                b = wrapping.gram_schmidt_projection_step(b, inner_product, basis_functions[i], transpose)
            norm_b = sqrt(transpose(b)*inner_product*b)
            if norm_b != 0.:
                b /= norm_b
            basis_functions[n_basis - 1] = b
    return _GramSchmidt
