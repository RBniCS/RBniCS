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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from ufl.core.operator import Operator
from dolfin import assemble
from RBniCS.backends.basic import transpose as basic_transpose
import RBniCS.backends.fenics
from RBniCS.backends.fenics.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.vector import Vector
import RBniCS.backends.fenics.wrapping
import RBniCS.backends.numpy
from RBniCS.utils.decorators import backend_for

@backend_for("fenics", online_backend="numpy", inputs=((BasisFunctionsMatrix, Form, Function.Type(), FunctionsList, Operator, Vector.Type()), ))
def transpose(arg):
    def AdditionalIsVector(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 1
    def AdditionalIsMatrix(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 2
    return basic_transpose(arg, RBniCS.backends.fenics, RBniCS.backends.fenics.wrapping, RBniCS.backends.numpy, AdditionalFunctionTypes=(Operator, ), AdditionalIsVector=AdditionalIsVector, AdditionalIsMatrix=AdditionalIsMatrix)
    
        