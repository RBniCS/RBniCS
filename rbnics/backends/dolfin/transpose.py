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
import rbnics.backends.dolfin
from rbnics.backends.dolfin.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_from_ufl_operators
from rbnics.utils.decorators import backend_for

@backend_for("dolfin", inputs=((BasisFunctionsMatrix, Form, Function.Type(), FunctionsList, Operator, Vector.Type()), ))
def transpose(arg):
    def AdditionalIsFunction(arg):
        return isinstance(arg, Operator)
    def ConvertAdditionalFunctionTypes(arg):
        assert isinstance(arg, Operator)
        return function_from_ufl_operators(arg)
    def AdditionalIsVector(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 1
    def ConvertAdditionalVectorTypes(arg):
        assert isinstance(arg, Form) and len(arg.arguments()) is 1
        return assemble(arg)
    def AdditionalIsMatrix(arg):
        return isinstance(arg, Form) and len(arg.arguments()) is 2
    def ConvertAdditionalMatrixTypes(arg):
        assert isinstance(arg, Form) and len(arg.arguments()) is 2
        return assemble(arg)
    return basic_transpose(arg, rbnics.backends.dolfin, rbnics.backends.dolfin.wrapping, AdditionalIsFunction, ConvertAdditionalFunctionTypes, AdditionalIsVector, ConvertAdditionalVectorTypes, AdditionalIsMatrix, ConvertAdditionalMatrixTypes)
    
        
