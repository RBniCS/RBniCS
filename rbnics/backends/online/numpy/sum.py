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

from rbnics.backends.online.basic import sum as basic_sum
from rbnics.backends.online.numpy.product import ProductOutput
from rbnics.utils.decorators import backend_for

# product function to assemble truth/reduced affine expansions. To be used in combination with product,
# even though product actually carries out both the sum and the product!
@backend_for("numpy", inputs=(ProductOutput, ))
def sum(product_output):
    return basic_sum(product_output)
