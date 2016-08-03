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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.linear_algebra.truth_function import TruthFunction
from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_function import OnlineFunction
from RBniCS.linear_algebra.online_vector import OnlineVector_Type
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Auxiliary transpose method to be used in RBniCS. See functions_list.py and compute_scalar_product.py for more details.
def transpose(arg):
    from RBniCS.linear_algebra.functions_list import FunctionsList, FunctionsList_Transpose
    assert (
        isinstance(arg, FunctionsList) 
            or
        isinstance(arg, TruthVector) or isinstance(arg, OnlineVector_Type)
            or
        isinstance(arg, TruthFunction) or isinstance(arg, OnlineFunction)
    )
    if isinstance(arg, FunctionsList):
        return FunctionsList_Transpose(arg)
    elif (
        isinstance(arg, TruthVector) or isinstance(arg, OnlineVector_Type) 
            or
        isinstance(arg, TruthFunction) or isinstance(arg, OnlineFunction)
    ):
        return Vector_Transpose(arg)
    else: # impossible to arrive here anyway, thanks to the assert
        raise TypeError("Invalid arguments in transpose.")
        
# Auxiliary class: transpose of a vector
class Vector_Transpose(object):
    def __init__(self, vector):
        assert (
            isinstance(vector, TruthVector) or isinstance(vector, OnlineVector_Type)
                or
            isinstance(vector, TruthFunction) or isinstance(vector, OnlineFunction)
        )
        if isinstance(vector, TruthVector) or isinstance(vector, OnlineVector_Type):
            self.vector = vector
        elif isinstance(vector, TruthFunction) or isinstance(vector, OnlineFunction):
            self.vector = vector.vector()
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in Vector_Transpose.__init__.")
    
    def __mul__(self, matrixOrVector): # self * matrixOrVector
        assert (
            isinstance(matrixOrVector, TruthMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type)
                or
            isinstance(matrixOrVector, TruthVector) or isinstance(matrixOrVector, OnlineVector_Type)
                or
            isinstance(matrixOrVector, TruthFunction) or isinstance(matrixOrVector, OnlineFunction)
        )
        if isinstance(matrixOrVector, TruthMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type):
            return Vector_Transpose__times__Matrix(self.vector, matrixOrVector)
        elif isinstance(matrixOrVector, TruthVector) or isinstance(matrixOrVector, TruthFunction):
            if isinstance(matrixOrVector, TruthFunction):
                matrixOrVector = matrixOrVector.vector()
            return self.vector.inner(matrixOrVector)
        elif isinstance(matrixOrVector, OnlineVector_Type) or isinstance(matrixOrVector, OnlineFunction):
            if isinstance(matrixOrVector, OnlineFunction):
                matrixOrVector = matrixOrVector.vector()
            output = self.vector.T*matrixOrVector
            assert output.shape == (1, 1)
            return output.item(0, 0)
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in Vector_Transpose.__mul__.")
              
# Auxiliary class: multiplication of the transpose of a Vector with a Matrix
class Vector_Transpose__times__Matrix(object):
    def __init__(self, vector, matrix):
        assert isinstance(vector, TruthVector) or isinstance(vector, OnlineVector_Type)
        assert isinstance(matrix, TruthMatrix) or isinstance(matrix, OnlineMatrix_Type)
        self.vector = vector
        self.matrix = matrix
          
    # self * vector2
    def __mul__(self, vector2):
        assert (
            isinstance(vector2, TruthVector) or isinstance(vector2, OnlineVector_Type)
                or
            isinstance(vector2, TruthFunction) or isinstance(vector2, OnlineFunction)
        )
        if isinstance(vector2, TruthVector) or isinstance(vector2, TruthFunction):
            if isinstance(vector2, TruthFunction):
                vector2 = vector2.vector()
            return self.vector.inner(self.matrix*vector2)
        elif isinstance(vector2, OnlineVector_Type) or isinstance(vector2, OnlineFunction):
            if isinstance(vector2, OnlineFunction):
                vector2 = vector2.vector()
            output = self.vector.T*(self.matrix*vector2)
            assert output.shape == (1, 1)
            return output.item(0, 0)
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in Vector_Transpose__times__Matrix.__mul__.")
     
#  @}
########################### end - OFFLINE STAGE - end ########################### 

