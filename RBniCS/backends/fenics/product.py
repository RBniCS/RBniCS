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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import DirichletBC, Form
from RBniCS.backend.fenics.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.fenics.matrix import Matrix_Type
from RBniCS.backends.fenics.vector import Vector_Type
from RBniCS.utils.decorators import any, backend_for

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("FEniCS", inputs=(tuple, AffineExpansionStorage, any(tuple, None)))
def product(thetas, operators, thetas2=None):
    assert thetas2 is None
    assert len(thetas) == len(operators)
    if operators.type() is DirichletBC: 
        # Detect BCs defined on the same boundary
        combined = dict() # from (function space, boundary) to value
        for (op_index, op) in enumerate(operators):
            for bc in op:
                key = (bc.function_space, bc.subdomains, bc.subdomain_id)
                if not key in combined:
                    combined[key] = list()
                combined[key].append((bc.value, op_index))
        # Sum them
        output = list()
        for (key, item) in combined.iteritems():
            value = 0
            for addend in item:
                value += Constant(thetas[ addend[1] ]) * addend[0]
            output.append(DirichletBC(key[0], value, key[1], key[2]))
        return ProductOutput(output)
    elif operators.type() is Form:
        assert isinstance(operators[0], Matrix_Type) or isinstance(operators[0], Vector_Type)
        # Carry out the dot product (with respect to the index q over the affine expansion)
        if isinstance(operators[0], Matrix_Type):
            output = operators[0].copy()
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                output += theta*operator
            return ProductOutput(output)
        elif isinstance(operators[0], Vector_Type):
            output = operators[0].copy()
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                output.add_local(theta*operator.array())
            output.apply("add")
            return ProductOutput(output)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("product(): invalid operands.")
    else:
        raise AssertionError("product(): invalid operands.")
        
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
    
