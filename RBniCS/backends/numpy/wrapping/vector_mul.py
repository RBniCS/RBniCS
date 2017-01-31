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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import RBniCS.backends # avoid circular imports when importing numpy backend

def vector_mul_vector(vector1, vector2):
    if isinstance(vector1, RBniCS.backends.numpy.Function.Type()):
        vector1 = vector1.vector()
    if isinstance(vector2, RBniCS.backends.numpy.Function.Type()):
        vector2 = vector2.vector()
    output = vector1.T*vector2
    assert output.shape == (1, 1)
    return output.item(0, 0)

