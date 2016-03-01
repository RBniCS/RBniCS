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

###########################     OFFLINE AND ONLINE COMMON INTERFACES     ########################### 
## @defgroup OfflineOnlineInterfaces Common interfaces for offline and online
#  @{

from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector_Type as OnlineVector
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type as OnlineMatrix

# product function to assemble truth/reduced affine expansions. To be used in combination with python's sum.
def product(thetas, operators):
    output = []
    assert len(thetas) == len(operators)
    if \
        isinstance(operators[0], TruthMatrix) or isinstance(operators[0], TruthVector) \
    or \
        isinstance(operators[0], OnlineMatrix) or isinstance(operators[0], OnlineVector) \
    :
        for i in range(len(thetas)):
            output.append(thetas[i]*operators[i])
    elif isinstance(operators[0], list): # we use this Dirichlet BCs with FEniCS
        for i in range(len(output)):
            # Each element of the list contains a tuple. Owing to FEniCS documentation, its second argument is the function, to be multiplied by theta
            output_i = []
            for j in range(len(operators[i])):
                operators_i_j_list = list(operators[i][j])
                operators_i_j_list[1] = Constant(thetas[i])*operators_i_list[1]
                output_i.append(tuple(operators_i_j_list))
            output.append(output_i)
    else:
        raise RuntimeError("product(): invalid operands.")
    return output
        
#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 
