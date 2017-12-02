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

from itertools import product as cartesian_product
from numbers import Number
from rbnics.backends.online.numpy.affine_expansion_storage import AffineExpansionStorage
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.function import Function
from rbnics.utils.decorators import backend_for, ThetaType

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("numpy", inputs=(ThetaType, AffineExpansionStorage, ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    order = operators.order()
    first_operator = None
    assert order in (1, 2)
    if order == 1: # vector storage of affine expansion online data structures (e.g. reduced matrix/vector expansions)
        first_operator = operators[0]
        assert isinstance(first_operator, (Matrix.Type(), Vector.Type(), Function.Type(), Number))
        assert thetas2 is None
        assert len(thetas) == len(operators)
        for (index, (theta, operator)) in enumerate(zip(thetas, operators)):
            if index == 0:
                output = theta*operator
            elif theta != 0.:
                output += theta*operator
    elif order == 2: # matrix storage of affine expansion online data structures (e.g. error estimation ff/af/aa products)
        first_operator = operators[0, 0]
        assert isinstance(first_operator, (Matrix.Type(), Vector.Type(), Number))
        assert thetas2 is not None
        # no checks here on the first dimension of operators should be equal to len(thetas), and
        # similarly that the second dimension should be equal to len(thetas2), because the
        # current operator interface does not provide a 2D len method
        for (i, j) in cartesian_product(range(len(thetas)), range(len(thetas2))):
            if i == 0 and j == 0:
                output = thetas[0]*operators[0, 0]*thetas2[0]
            elif thetas[i] != 0. and thetas2[j] != 0.:
                output += thetas[i]*operators[i, j]*thetas2[j]
    else:
        raise ValueError("product(): invalid operands.")
    # Return
    return ProductOutput(output)
    
        
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
