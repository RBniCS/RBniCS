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

from dolfin import Function
from rbnics.problems.navier_stokes import NavierStokesProblem

class NavierStokesTensor3Problem(NavierStokesProblem):
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NavierStokesProblem.__init__(self, V, **kwargs)
        
        # Placeholders for tensor3 assembly
        self._solution_placeholder_1 = Function(V)
        self._solution_placeholder_2 = Function(V)
        self._solution_placeholder_3 = Function(V)
        
    def _init_operators(self):
        NavierStokesProblem._init_operators(self)
        # Add tensor3 version of nonlinear terms
        for term in ("c_tensor3", "dc_tensor3"):
            if term not in self.operator: # init was not called already
                self.operator[term] = self.assemble_operator(term) 
                # note that we do not store this in a AffineExpansionStorage, as no assembly should be done
