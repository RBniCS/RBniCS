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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector_Type, OnlineVector
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type, OnlineMatrix
from RBniCS.linear_algebra.compute_scalar_product import Vector_Transpose

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Type for storing a list of FE functions. From the user point of view this is
# the same as a matrix. Indeed, given a TruthMatrix A, a TruthVector F 
# and a FunctionsList Z, overriding __mul__ and __rmul__ operators
# allow to write expressions like transpose(Z)*A*Z and transpose(Z)*F
class FunctionsList(object):
    def __init__(self):
        self._list = []
    
    def enrich(self, functions):
        from dolfin import Function
        if isinstance(functions, Function): # one function
            self._list.append(functions.vector().copy()) # copy it explicitly
        else: # more than one function
            self._list.extend(functions) # assume that they where already copied
            
    def load(self, directory, filename):
        if self._list: # avoid loading multiple times
            return False
        if io_utils.exists_pickle_file(directory, filename):
            self._list = io_utils.load_pickle_file(directory, filename)
            return True
        else:
            return False
        
    def save(self, directory, filename):
        io_utils.save_pickle_file(self._list, directory, filename)
        for f in range(len(self._list)):
            full_filename = directory + "/" + filename + "_" + str(f) + ".pvd"
            if not os.path.exists(full_filename):
                file = File(filename, "compressed")
                file << self._list[f]
         
    def __getitem__(self, key):
        return self._list[key]
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)
    
    # self * onlineMatrixOrVector [used e.g. to compute Z*u_N or S*eigv]
    def __mul__(self, onlineMatrixOrVector):
        assert isinstance(onlineMatrixOrVector, OnlineMatrix_Type) or isinstance(onlineMatrixOrVector, OnlineVector_Type)
        if isinstance(onlineMatrixOrVector, OnlineMatrix_Type):
            output = FunctionsList()
            dim = onlineMatrixOrVector.shape[1]
            for i in range(dim):
                output_i = self._list[0]*onlineMatrixOrVector[i, 0]
                for j in range(1, len(self._list)):
                    output_i += self._list[j]*onlineMatrixOrVector[i, j]
                output.enrich(output_i)
        elif isinstance(onlineMatrixOrVector, OnlineVector_Type):
            output = self._list[0]*onlineMatrixOrVector.item(0)
            for j in range(1, len(self._list)):
                output.add_local(self._list[j].array()*onlineMatrixOrVector.item(j))
            output.apply("add")
            return output
        else: # impossible to arrive here anyway, thanks to the assert
            raise RuntimeError("Invalid arguments in FunctionsList.__mul__.")
        
# Auxiliary class: transpose of a FunctionsList
class FunctionsList_Transpose(object):
    def __init__(self, functionsList):
        assert isinstance(functionsList, FunctionsList)
        self.functionsList = functionsList
    
    # self * truthMatrixOrVector [used e.g. to compute Z^T*F]
    def __mul__(self, truthMatrixOrVector):
        assert isinstance(truthMatrixOrVector, TruthMatrix) or isinstance(truthMatrixOrVector, TruthVector)
        if isinstance(truthMatrixOrVector, TruthMatrix):
            return FunctionsList_Transpose__times__TruthMatrix(self.functionsList, truthMatrixOrVector)
        elif isinstance(truthMatrixOrVector, TruthVector):
            dim = len(self.functionsList)
            onlineVector = OnlineVector(dim)
            for i in range(dim):
                onlineVector[i] = Vector_Transpose(self.functionsList[i])*truthMatrixOrVector
            return onlineVector
        else: # impossible to arrive here anyway, thanks to the assert
            raise RuntimeError("Invalid arguments in FunctionsList_Transpose.__mul__.")

# Auxiliary class: multiplication of the transpose of a functions list with a TruthMatrix
class FunctionsList_Transpose__times__TruthMatrix(object):
    def __init__(self, functionsList, truthMatrix):
        assert isinstance(functionsList, FunctionsList)
        self.functionsList = functionsList
        self.truthMatrix = truthMatrix
      
    # self * functionsList2 [used e.g. to compute Z^T*A*Z or S^T*X*S]
    def __mul__(self, functionsList2):
        assert isinstance(functionsList2, FunctionsList)
        assert len(self.functionsList) == len(functionsList2)
        dim = len(self.functionsList)
        onlineMatrix = OnlineMatrix(dim, dim)
        for i in range(dim):
            for j in range(dim):
                onlineMatrix[i, j] = Vector_Transpose(self.functionsList[i])*self.truthMatrix*functionsList2[j]
        return onlineMatrix
     
#  @}
########################### end - OFFLINE STAGE - end ########################### 

