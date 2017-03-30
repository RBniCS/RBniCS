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

from ufl import Form
from ufl.core.operator import Operator
from dolfin import assemble
from rbnics.backends.basic import transpose as basic_transpose
import rbnics.backends.fenics
from rbnics.backends.fenics.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.fenics.function import Function
from rbnics.backends.fenics.functions_list import FunctionsList
from rbnics.backends.fenics.vector import Vector
import rbnics.backends.fenics.wrapping
import rbnics.backends.numpy
from rbnics.utils.decorators import backend_for

@backend_for("fenics", online_backend="numpy", inputs=((BasisFunctionsMatrix, Form, Function.Type(), FunctionsList, Operator, Vector.Type()), ))
def transpose(arg):
    def AdditionalIsVector(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 1
    def AdditionalIsMatrix(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 2
    return basic_transpose(arg, rbnics.backends.fenics, rbnics.backends.fenics.wrapping, rbnics.backends.numpy, AdditionalFunctionTypes=(Operator, ), AdditionalIsVector=AdditionalIsVector, AdditionalIsMatrix=AdditionalIsMatrix)
    
        
