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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from ufl.core.operator import Operator
from dolfin import assemble
import RBniCS.backends # avoid circular imports when importing fenics backend
from RBniCS.backends.fenics.wrapping import function_from_ufl_operators

def vector_mul_vector(vector1, vector2):
    if isinstance(vector1, (RBniCS.backends.fenics.Function.Type(), Operator)):
        vector1 = function_from_ufl_operators(vector1).vector()
    elif isinstance(vector1, Form):
        assert len(vector1.arguments()) is 1
        vector1 = assemble(vector1)
    if isinstance(vector2, (RBniCS.backends.fenics.Function.Type(), Operator)):
        vector2 = function_from_ufl_operators(vector2).vector()
    elif isinstance(vector2, Form):
        assert len(vector2.arguments()) is 1
        vector2 = assemble(vector2)
    return vector1.inner(vector2)

