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

from numbers import Number
from ufl import Form
from ufl.core.operator import Operator
from dolfin import assemble, Constant, Expression, project
from rbnics.backends.dolfin.affine_expansion_storage import AffineExpansionStorage_Base, AffineExpansionStorage_DirichletBC, AffineExpansionStorage_Form, AffineExpansionStorage_Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_copy, tensor_copy
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC, ProductOutputDirichletBC
from rbnics.utils.decorators import backend_for, ComputeThetaType, list_of, overload

# Need to customize ThetaType in order to also include dolfin' ParametrizedConstant (of type Expression), which is a side effect of DEIM decorator
ThetaType = ComputeThetaType((Expression, Operator))

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("dolfin", inputs=(ThetaType, AffineExpansionStorage_Base, None))
def product(thetas, operators, thetas2=None):
    assert len(thetas) == len(operators)
    return _product(thetas, operators)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage_DirichletBC):
    # Detect BCs defined on the same boundary
    combined = dict() # from (function space, boundary) to value
    for (op_index, op) in enumerate(operators):
        for bc in op:
            key = bc.identifier()
            if key not in combined:
                combined[key] = list()
            combined[key].append((bc, op_index))
    # Sum them
    output = ProductOutputDirichletBC()
    for (key, item) in combined.items():
        value = 0
        for addend in item:
            theta = float(thetas[addend[1]])
            fun = addend[0].value()
            value += Constant(theta)*fun
        V = item[0][0].function_space()
        if len(V.component()) == 0: # FunctionSpace
            value = project(value, V)
        else: # subspace of a FunctionSpace
            value = project(value, V.collapse())
        args = list()
        args.append(V)
        args.append(value)
        args.extend(item[0][0].domain_args)
        args.extend(item[0][0]._sorted_kwargs)
        output.append(DirichletBC(*args))
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage_Form):
    return _product(thetas, operators._content)

@overload
def _product(thetas: ThetaType, operators: list_of(Form)):
    # Keep the operators as Forms and delay assembly as long as possible
    output = 0
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output += Constant(theta)*operator
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: list_of(Matrix.Type())):
    output = tensor_copy(operators[0])
    output.zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output += theta*operator
    return ProductOutput(output)

@overload
def _product(thetas: ThetaType, operators: list_of(Vector.Type())):
    output = tensor_copy(operators[0])
    output.zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output.add_local(theta*operator.array())
    output.apply("add")
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: list_of(Number)):
    output = 0.
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output += theta*operator
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: (list_of((Form, Matrix.Type())), list_of((Form, Vector.Type())))):
    # Since there are both forms and matrices/vectors among provided operators,
    # we are forced to assemble every form in order to sum them
    # with the other matrices/vectors
    assembled_operators = list()
    for operator in operators:
        if isinstance(operator, Form):
            assembled_operators.append(assemble(operator, keep_diagonal=True))
        else:
            assembled_operators.append(operator)
    return _product(thetas, assembled_operators)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage_Function):
    output = function_copy(operators[0])
    output.vector().zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output.vector().add_local(theta*operator.vector().array())
    output.vector().apply("add")
    return ProductOutput(output)
    
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
