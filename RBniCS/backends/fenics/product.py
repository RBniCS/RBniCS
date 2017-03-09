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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from dolfin import assemble, Constant, DirichletBC, project
from RBniCS.backends.fenics.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.wrapping import function_copy, tensor_copy
from RBniCS.utils.decorators import backend_for, ComputeThetaType
from RBniCS.utils.mpi import log, PROGRESS

# Need to customize ThetaType in order to also include FEniCS' Constant, which is a side effect of DEIM decorator
ThetaType = ComputeThetaType((Constant, ))

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("fenics", inputs=(ThetaType, AffineExpansionStorage, ThetaType + (None, )))
def product(thetas, operators, thetas2=None):
    assert thetas2 is None
    assert len(thetas) == len(operators)
    if operators.type() == "AssembledForm":
        assert isinstance(operators[0], (Matrix.Type(), Vector.Type()))
        # Carry out the dot product (with respect to the index q over the affine expansion)
        if isinstance(operators[0], Matrix.Type()):
            output = tensor_copy(operators[0])
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                theta = float(theta)
                output += theta*operator
            return ProductOutput(output)
        elif isinstance(operators[0], Vector.Type()):
            output = tensor_copy(operators[0])
            output.zero()
            for (theta, operator) in zip(thetas, operators):
                theta = float(theta)
                output.add_local(theta*operator.array())
            output.apply("add")
            return ProductOutput(output)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("product(): invalid operands.")
    elif operators.type() == "DirichletBC": 
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
            try:
                dirichlet_bc = DirichletBC(key[0], value, key[1], key[2])
            except RuntimeError: # key[0] was a subspace, and DirichletBC does not handle collapsing
                V_collapsed = key[0].collapse()
                value_projected_collapsed = project(value, V_collapsed)
                dirichlet_bc = DirichletBC(key[0], value_projected_collapsed, key[1], key[2])
            output.append(dirichlet_bc)
        return ProductOutput(output)
    elif operators.type() == "Form":
        log(PROGRESS, "re-assemblying form (due to inefficient evaluation)")
        assert isinstance(operators[0], Form)
        output = 0
        for (theta, operator) in zip(thetas, operators):
            output += Constant(theta)*operator
        # keep_diagonal is enabled because it is needed to constrain DirichletBC eigenvalues in SCM
        output = assemble(output, keep_diagonal=True)
        return ProductOutput(output)
    elif operators.type() == "Function":
        output = function_copy(operators[0])
        output.vector().zero()
        for (theta, operator) in zip(thetas, operators):
            theta = float(theta)
            output.vector().add_local(theta*operator.vector().array())
        output.vector().apply("add")
        return ProductOutput(output)
    else:
        raise AssertionError("product(): invalid operands.")
        
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
    
