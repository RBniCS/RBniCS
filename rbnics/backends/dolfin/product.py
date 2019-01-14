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

from numbers import Number
from ufl import Form
from ufl.core.operator import Operator
from dolfin import Constant, Expression
from rbnics.backends.dolfin.affine_expansion_storage import AffineExpansionStorage_Base, AffineExpansionStorage_DirichletBC, AffineExpansionStorage_Form, AffineExpansionStorage_Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.non_affine_expansion_storage import NonAffineExpansionStorage
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_copy, tensor_copy
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC, ProductOutputDirichletBC
from rbnics.eim.utils.decorators import add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import backend_for, ComputeThetaType, overload, tuple_of

# Need to customize ThetaType in order to also include dolfin' ParametrizedConstant (of type Expression), which is a side effect of DEIM decorator:
# this is the reason why in the following theta coefficients are preprocessed by float().
ThetaType = ComputeThetaType((Expression, Operator))

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("dolfin", inputs=(ThetaType, (AffineExpansionStorage_Base, NonAffineExpansionStorage), None))
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
        value = function_copy(item[0][0].value())
        value.vector().zero()
        for addend in item:
            theta = float(thetas[addend[1]])
            fun = addend[0].value()
            value.vector().add_local(theta*fun.vector().get_local())
        value.vector().apply("add")
        args = list()
        args.append(item[0][0].function_space())
        args.append(value)
        args.extend(item[0][0]._domain)
        output.append(DirichletBC(*args, **item[0][0]._kwargs))
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: (AffineExpansionStorage_Form, NonAffineExpansionStorage)):
    return _product(thetas, operators._content)
    
@overload
def _product(thetas: ThetaType, operators: tuple_of(Form)):
    try:
        output = _product_forms_output_cache[operators]
    except KeyError:
        # Keep the operators as Forms and delay assembly as long as possible
        output = 0
        constants = list()
        for (theta, operator) in zip(thetas, operators):
            theta = float(theta)
            constant = Constant(theta)
            output += constant*operator
            constants.append(constant)
        output = ProductOutput(output)
        _product_forms_output_cache[operators] = output
        _product_forms_constants_cache[operators] = constants
        return output
    else:
        constants = _product_forms_constants_cache[operators]
        for (theta, constant) in zip(thetas, constants):
            theta = float(theta)
            constant.assign(theta)
        return output
_product_forms_output_cache = Cache()
_product_forms_constants_cache = Cache()
    
@overload
def _product(thetas: ThetaType, operators: tuple_of(ParametrizedTensorFactory)):
    operators_as_forms = tuple(operator._form for operator in operators)
    try:
        output = _product_parametrized_tensor_factories_output_cache[operators_as_forms]
    except KeyError:
        # Keep the operators as ParametrizedTensorFactories and delay assembly as long as possible
        output = _product(thetas, operators_as_forms)
        output = ParametrizedTensorFactory(output.sum_product_return_value)
        problems = [get_problem_from_parametrized_operator(operator) for operator in operators]
        assert all([problem is problems[0] for problem in problems])
        add_to_map_from_parametrized_operator_to_problem(output, problems[0])
        output = ProductOutput(output)
        _product_parametrized_tensor_factories_output_cache[operators_as_forms] = output
        _product_parametrized_tensor_factories_constants_cache[operators_as_forms] = _product_forms_constants_cache[operators_as_forms]
        return output
    else:
        constants = _product_parametrized_tensor_factories_constants_cache[operators_as_forms]
        for (theta, constant) in zip(thetas, constants):
            theta = float(theta)
            constant.assign(theta)
        return output
_product_parametrized_tensor_factories_output_cache = Cache()
_product_parametrized_tensor_factories_constants_cache = Cache()
    
@overload
def _product(thetas: ThetaType, operators: tuple_of(Matrix.Type())):
    output = tensor_copy(operators[0])
    output.zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output += theta*operator
    return ProductOutput(output)

@overload
def _product(thetas: ThetaType, operators: tuple_of(Vector.Type())):
    output = tensor_copy(operators[0])
    output.zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output.add_local(theta*operator.get_local())
    output.apply("add")
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: tuple_of(Number)):
    output = 0.
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output += theta*operator
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage_Function):
    output = function_copy(operators[0])
    output.vector().zero()
    for (theta, operator) in zip(thetas, operators):
        theta = float(theta)
        output.vector().add_local(theta*operator.vector().get_local())
    output.vector().apply("add")
    return ProductOutput(output)
    
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
