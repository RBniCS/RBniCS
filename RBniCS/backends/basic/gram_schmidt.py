# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file gram_schmidt.py
#  @brief Implementation of the Gram Schmidt process
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from RBniCS.backend.abstract import GramSchmidt as AbstractGramSchmidt
from RBniCS.utils.decorators import Extends, override

@Extends(AbstractFunctionsList)
class GramSchmidt(AbstractFunctionsList):
    @override
    def __init__(self, X, backend, wrapping):
        # Inner product
        self.X = X
        self.backend = backend
        self.wrapping = wrapping
        
    @override
    def apply(self, Z):
        assert len(self.X) == 1 # note that we cannot move this assert in __init__ because
                                # self.X has not been assembled yet there
        X = self.X[0]
        
        transpose = self.backend.transpose

        n_basis = len(Z) # basis are store as a list of vectors
        b = Z[n_basis - 1] # reference to the last basis
        for i in range(n_basis - 1):
            self.wrapping.gram_schimdt_projection_step(b, X, Z[i])
        b.vector()[:] /= sqrt(transpose(b)*X*b)
        
