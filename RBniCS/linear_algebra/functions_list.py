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

from RBniCS.io_utils.exportable_list import ExportableList
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
class FunctionsList(ExportableList):
    def __init__(self, original_list=None):
        ExportableList.__init__(self, "pickle", original_list)
        self._precomputed_slices = dict() # from tuple to AffineExpansionOnlineStorage
    
    def enrich(self, functions):
        from dolfin import Function, GenericVector
        if isinstance(functions, Function): # one function
            self._list.append(functions.vector().copy()) # copy it explicitly
        elif isinstance(functions, GenericVector): # one function
            self._list.append(functions.copy()) # copy it explicitly
        else: # more than one function
            self._list.extend(functions) # assume that they where already copied
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    def append(self, functions):
        import warnings
        warnings.warn("Please use the enrich() method that provides a more self explanatory name.")
        self.enrich(functions)
        
    def load(self, directory, filename, V):
        if self._list: # avoid loading multiple times
            return False
        with open(directory + "/" + filename + ".length", "r") as length:
            Nmax = int(length.readline())
        from dolfin import File, Function
        fun = Function(V)
        for f in range(Nmax):
            full_filename = directory + "/" + filename + "_" + str(f) + ".xml"
            file = File(full_filename)
            file >> fun
            self.enrich(fun)
        return True
        
    def save(self, directory, filename, V):
        with open(directory + "/" + filename + ".length", "w") as length:
            length.write(str(len(self._list)))
        from dolfin import File, Function
        for f in range(len(self._list)):
            list_f = Function(V, self._list[f])
            full_filename = directory + "/" + filename + "_" + str(f) + ".pvd"
            file = File(full_filename, "compressed")
            file << list_f
            full_filename = directory + "/" + filename + "_" + str(f) + ".xml"
            file = File(full_filename)
            file << list_f
    
    # self * onlineMatrixOrVector [used e.g. to compute Z*u_N or S*eigv]
    def __mul__(self, onlineMatrixOrVector):
        assert isinstance(onlineMatrixOrVector, OnlineMatrix_Type) or isinstance(onlineMatrixOrVector, OnlineVector_Type)
        if isinstance(onlineMatrixOrVector, OnlineMatrix_Type):
            output = FunctionsList()
            dim = onlineMatrixOrVector.shape[1]
            for j in range(dim):
                assert len(onlineMatrixOrVector[:, j]) == len(self._list)
                output_j = self._list[0]*onlineMatrixOrVector[0, j]
                for i in range(1, len(self._list)):
                    output_j.add_local(self._list[i].array()*onlineMatrixOrVector[i, j])
                output_j.apply("add")
                output.enrich(output_j)
            return output
        elif isinstance(onlineMatrixOrVector, OnlineVector_Type):
            assert len(onlineMatrixOrVector) == len(self._list)
            output = self._list[0]*onlineMatrixOrVector.item(0)
            for i in range(1, len(self._list)):
                output.add_local(self._list[i].array()*onlineMatrixOrVector.item(i))
            output.apply("add")
            return output
        else: # impossible to arrive here anyway, thanks to the assert
            raise RuntimeError("Invalid arguments in FunctionsList.__mul__.")
            
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None and key.step is None
            if key.stop in self._precomputed_slices:
                return self._precomputed_slices[key.stop]
                            
            assert key.stop <= len(self._list)            
            if key.stop == len(self._list):
                self._precomputed_slices[key.stop] = self
                return self
            
            output = FunctionsList(self._list[key])
            self._precomputed_slices[key.stop] = output
            return output
                
        else: # return the element at position "key" in the storage
            return self._list[key]
        
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
      
    # self * functionsList2 [used e.g. to compute Z^T*A*Z or S^T*X*S (return OnlineMatrix), or Riesz_A^T*X*Riesz_F (return OnlineVector)]
    def __mul__(self, functionsList2OrTruthVector):
        assert isinstance(functionsList2OrTruthVector, FunctionsList) or isinstance(functionsList2OrTruthVector, TruthVector)
        if isinstance(functionsList2OrTruthVector, FunctionsList):
            assert len(self.functionsList) == len(functionsList2OrTruthVector)
            dim = len(self.functionsList)
            onlineMatrix = OnlineMatrix(dim, dim)
            for j in range(dim):
                matrixTimesVectorj = self.truthMatrix*functionsList2OrTruthVector[j]
                for i in range(dim):
                    onlineMatrix[i, j] = Vector_Transpose(self.functionsList[i])*matrixTimesVectorj
            return onlineMatrix
        else:
            dim = len(self.functionsList)
            onlineVector = OnlineVector(dim)
            matrixTimesVector = self.truthMatrix*functionsList2OrTruthVector
            for i in range(dim):
                onlineVector[i] = Vector_Transpose(self.functionsList[i])*matrixTimesVector
            return onlineVector
     
#  @}
########################### end - OFFLINE STAGE - end ########################### 

