# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.backends.common.affine_expansion_storage import AffineExpansionStorage
from rbnics.utils.decorators import backend_for, overload, ThetaType

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("common", inputs=(ThetaType, AffineExpansionStorage, ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    return _product(thetas, operators, thetas2)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage, thetas2: None):
    output = 0.
    assert len(thetas) == len(operators)
    for (theta, operator) in zip(thetas, operators):
        output += theta*operator
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage, thetas2: ThetaType):
    output = 0.
    # no checks here on the first dimension of operators should be equal to len(thetas), and
    # similarly that the second dimension should be equal to len(thetas2), because the
    # current operator interface does not provide a 2D len method
    for i, theta_i in enumerate(thetas):
        for j, theta2_j in enumerate(thetas2):
            output += theta_i*operators[i, j]*theta2_j
    return ProductOutput(output)
    
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
