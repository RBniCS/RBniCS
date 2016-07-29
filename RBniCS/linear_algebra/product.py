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

from dolfin import Constant
from RBniCS.linear_algebra.affine_expansion_offline_storage import AffineExpansionOfflineStorage, _AssembledFormsAffineExpansionOfflineStorageContent, _DirichletBCsAffineExpansionOfflineStorageContent
from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.affine_expansion_online_storage import AffineExpansionOnlineStorage
from RBniCS.linear_algebra.online_vector import OnlineVector_Type
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type

# product function to assemble truth/reduced affine expansions. To be used in combination with sum.
def product(thetas, _operators, thetas2=None):
    if isinstance(_operators, AffineExpansionOfflineStorage) and not isinstance(_operators._content, AffineExpansionOnlineStorage):
        assert thetas2 is None
        operators = _operators._content
        assert len(thetas) == len(operators)
        if isinstance(operators, _DirichletBCsAffineExpansionOfflineStorageContent): 
            output = _DirichletBCsProductOutput()
            for i in range(len(thetas)):
                # Each element of the list contains a tuple. Owing to FEniCS documentation, its second argument is the function, to be multiplied by theta
                output_i = list()
                for j in range(len(operators[i])):
                    assert len(operators[i][j]) == 4
                    operators_i_j_list = list(operators[i][j])
                    operators_i_j_list[1] = Constant(thetas[i])*operators_i_j_list[1]
                    output_i.append(tuple(operators_i_j_list))
                output.append(output_i)
            return output
        elif isinstance(operators, _AssembledFormsAffineExpansionOfflineStorageContent):
            assert isinstance(operators[0], TruthMatrix) or isinstance(operators[0], TruthVector)
            # Carry out the dot product (with respect to the index q over the affine expansion)
            if isinstance(operators[0], TruthMatrix):
                output = operators[0].copy()
                output.zero()
                for i in range(len(thetas)):
                    output += thetas[i]*operators[i]
                return _DotProductOutput(output)
            elif isinstance(operators[0], TruthVector):
                output = operators[0].copy()
                output.zero()
                for i in range(len(thetas)):
                    output.add_local(thetas[i]*operators[i].array())
                output.apply("insert")
                return _DotProductOutput(output)
            else: # impossible to arrive here anyway thanks to the assert
                raise TypeError("product(): invalid operands.")
        else:
            raise TypeError("product(): invalid operands.")
    elif \
        (isinstance(_operators, AffineExpansionOfflineStorage) and isinstance(_operators._content, AffineExpansionOnlineStorage)) \
            or \
        isinstance(_operators, AffineExpansionOnlineStorage) \
    :
        if \
            isinstance(_operators, AffineExpansionOfflineStorage) and isinstance(_operators._content, AffineExpansionOnlineStorage) \
        :
            operators = _operators._content
        else: # isinstance(_operators, AffineExpansionOnlineStorage)
            operators = _operators
        order = operators.order()
        if order == 1: # vector storage of affine expansion online data structures (e.g. reduced matrix/vector expansions)
            assert isinstance(operators[0], OnlineMatrix_Type) or isinstance(operators[0], OnlineVector_Type)
            assert thetas2 is None
            assert len(thetas) == len(operators)
            # Single for loop version:
            output = thetas[0]*operators[0]
            for i in range(1, len(thetas)):
                output += thetas[i]*operators._content[i]
            return _DotProductOutput(output)
            '''
            # Vectorized version:
            # Profiling has reveleaded that the following vectorized (over q) version
            # introduces an overhead of 10%~20%
            from numpy import asmatrix
            output = asmatrix(thetas)*operators.as_matrix().transpose()
            output = output.item(0, 0)
            return _DotProductOutput(output)
            '''
        elif order == 2: # matrix storage of affine expansion online data structures (e.g. error estimation ff/af/aa products)
            assert \
                isinstance(operators[0, 0], OnlineMatrix_Type) or \
                isinstance(operators[0, 0], OnlineVector_Type) or \
                isinstance(operators[0, 0], float)
            assert thetas2 is not None
            # no checks here on the first dimension of operators should be equal to len(thetas), and
            # similarly that the second dimension should be equal to len(thetas2), because the
            # current operator interface does not provide a 2D len method
            '''
            # Double for loop version:
            # Profiling has revelead a sensible speedup for large values of N and Q 
            # when compared to the double/triple/quadruple for loop in the legacy version.
            # Vectorized version provides an additional 25%~50% speedup when dealing with
            # the (A, A) Riesz representor products (case of quadruple loop),
            # while double for loop only introduces overhead when for (F, F) Riesz 
            # representor products (case of double loop).
            output = 0.
            for i in range(len(thetas)):
                for j in range(len(thetas2)):
                    output += thetas[i]*operators[i, j]*thetas2[j]
            # Thus we selected the following:
            '''
            # Vectorized version:
            from numpy import asmatrix
            thetas_vector = asmatrix(thetas)
            thetas2_vector = asmatrix(thetas2).transpose()
            output = thetas_vector*operators.as_matrix()*thetas2_vector
            return _DotProductOutput(output.item(0, 0))
        else:
            raise TypeError("product(): invalid operands.")
    else:
        raise TypeError("product(): invalid operands.")

# Auxiliary class to signal to the sum() function that the sum has already been performed by the dot product
class _DotProductOutput(list):
    def __init__(self, content):
        self.append(content)
        
# Auxiliary class to signal to the sum() function that it is dealing with an expansion of Dirichlet BCs
class _DirichletBCsProductOutput(list):
    pass
        
#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 
