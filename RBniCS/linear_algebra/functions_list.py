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

from dolfin import FunctionSpace
from RBniCS.utils.io import ExportableList, File, NumpyIO
from RBniCS.utils.mpi import mpi_comm
from RBniCS.utils.decorators import Extends, override
from RBniCS.linear_algebra.truth_function import TruthFunction
from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_function import OnlineFunction
from RBniCS.linear_algebra.online_vector import OnlineVector_Type, OnlineVector
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Type, OnlineMatrix
from RBniCS.linear_algebra.transpose import Vector_Transpose

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Type for storing a list of FE functions. From the user point of view this is
# the same as a matrix. Indeed, given a TruthMatrix A, a TruthVector F 
# and a FunctionsList Z, overriding __mul__ and __rmul__ operators
# allow to write expressions like transpose(Z)*A*Z and transpose(Z)*F
@Extends(ExportableList)
class FunctionsList(ExportableList):
    @override
    def __init__(self, V_or_Z, original_list=None):
        ExportableList.__init__(self, "pickle", original_list)
        assert (
            isinstance(V_or_Z, FunctionSpace) 
                or
            isinstance(V_or_Z, FunctionsList)
                    or
            V_or_Z is None
        )
        if isinstance(V_or_Z, FunctionSpace):
            self.V = V_or_Z
        elif (
                isinstance(V_or_Z, FunctionsList)
                    or
                V_or_Z is None # used internally in __mul__ and __getitem__
            ):
            self.V = None
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.__init__().")
        self._precomputed_slices = dict() # from tuple to AffineExpansionOnlineStorage
    
    def enrich(self, functions):
        def append(function):
            assert isinstance(function, TruthFunction) or isinstance(function, OnlineFunction)
            if isinstance(function, TruthFunction):
                assert (
                    self.V is not None 
                        and 
                    isinstance(self.V, FunctionSpace) 
                        and
                    function.function_space() == self.V
                )
                self._list.append(function.copy(deepcopy=True))
            elif isinstance(function, OnlineFunction):
                assert self.V is None
                self._list.append(function.copy(deepcopy=True))
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in FunctionsList.enrich().")
        if isinstance(functions, tuple) or isinstance(functions, list) or isinstance(functions, FunctionsList):
            for function in functions:
                append(function)
        else:
            append(functions)
        # Reset precomputed slices
        self._precomputed_slices = dict()
    
    @override
    def append(self, functions):
        import warnings
        warnings.warn("Please use the enrich() method that provides a more self explanatory name.")
        self.enrich(functions)
        
    def clear(self):
        self._list = list()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def load(self, directory, filename):
        if self._list: # avoid loading multiple times
            return False
        Nmax = None
        if mpi_comm.rank == 0:
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = mpi_comm.bcast(Nmax, root=0)
        if self.V is not None:
            assert isinstance(self.V, FunctionSpace)
            fun = TruthFunction(self.V)
            for index in range(Nmax):
                full_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
                file = File(full_filename)
                file >> fun
                self.enrich(fun)
        else:
            for index in range(Nmax):
                vec = NumpyIO.load_file(directory, filename + "_" + str(index))
                fun = OnlineFunction(vec)
                self.enrich(fun)
        return True
        
    @override
    def save(self, directory, filename):
        if mpi_comm.rank == 0:
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self._list)))
        if self.V is not None:
            assert isinstance(self.V, FunctionSpace)
            for (index, fun) in enumerate(self._list):
                full_filename = str(directory) + "/" + filename + "_" + str(index) + ".pvd"
                file = File(full_filename, "compressed")
                file << fun
                full_filename = str(directory) + "/" + filename + "_" + str(index) + ".xml"
                file = File(full_filename)
                file << fun
        else:
            for (index, fun) in enumerate(self._list):
                NumpyIO.save_file(fun.vector(), directory, filename + "_" + str(f))
    
    # self * onlineMatrixOrVector [used e.g. to compute Z*u_N or S*eigv]
    def __mul__(self, onlineMatrixOrVector):
        assert (
            isinstance(onlineMatrixOrVector, OnlineMatrix_Type)
                or
            isinstance(onlineMatrixOrVector, OnlineVector_Type)
                or
            isinstance(onlineMatrixOrVector, OnlineFunction)
        )
        if isinstance(onlineMatrixOrVector, OnlineMatrix_Type):
            output = FunctionsList(self.V)
            dim = onlineMatrixOrVector.shape[1]
            for j in range(dim):
                assert len(onlineMatrixOrVector[:, j]) == len(self._list)
                output_j = self._list[0].copy(deepcopy=True)
                if self.V is not None:
                    output_j.vector().zero()
                    for (i, fun_i) in enumerate(self._list):
                        output_j.vector().add_local(fun_i.vector().array()*onlineMatrixOrVector[i, j])
                    output_j.vector().apply("add")
                else:
                    output_j.vector()[:] = 0.
                    for (i, fun_i) in enumerate(self._list):
                        output_j.vector()[:] += fun_i.vector()*onlineMatrixOrVector[i, j]
                output.enrich(output_j)
            return output
        elif isinstance(onlineMatrixOrVector, OnlineVector_Type) or isinstance(onlineMatrixOrVector, OnlineFunction):
            if isinstance(onlineMatrixOrVector, OnlineFunction):
                onlineMatrixOrVector = onlineMatrixOrVector.vector()
            assert len(onlineMatrixOrVector) == len(self._list)
            output = self._list[0].copy(deepcopy=True)
            if self.V is not None:
                output.vector().zero()
                for (i, fun_i) in enumerate(self._list):
                    output.vector().add_local(fun_i.vector().array()*onlineMatrixOrVector.item(i))
                output.vector().apply("add")
            else:
                output.vector()[:] = 0.
                for (i, fun_i) in enumerate(self._list):
                    output.vector()[:] += fun_i.vector()*onlineMatrixOrVector.item(i)
            return output
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.__mul__.")
            
    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None and key.step is None
            if key.stop in self._precomputed_slices:
                return self._precomputed_slices[key.stop]
                            
            assert key.stop <= len(self._list)            
            if key.stop == len(self._list):
                self._precomputed_slices[key.stop] = self
                return self
            
            output = FunctionsList(self.V, self._list[key])
            self._precomputed_slices[key.stop] = output
            return output
                
        else: # return the element at position "key" in the storage
            return self._list[key]
            
    @override
    def __iter__(self):
        return self._list.__iter__()
        
# Auxiliary class: transpose of a FunctionsList
class FunctionsList_Transpose(object):
    def __init__(self, functionsList):
        assert isinstance(functionsList, FunctionsList)
        self.functionsList = functionsList
    
    # self * matrixOrVector [used e.g. to compute Z^T*F]
    def __mul__(self, matrixOrVector):
        assert (
            isinstance(matrixOrVector, TruthMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type)
                or
            isinstance(matrixOrVector, TruthVector) or isinstance(matrixOrVector, OnlineVector_Type)
        )
        if isinstance(matrixOrVector, TruthMatrix) or isinstance(matrixOrVector, OnlineMatrix_Type):
            return FunctionsList_Transpose__times__Matrix(self.functionsList, matrixOrVector)
        elif isinstance(matrixOrVector, TruthVector) or isinstance(matrixOrVector, OnlineVector_Type):
            dim = len(self.functionsList)
            onlineVector = OnlineVector(dim)
            for i in range(dim):
                onlineVector[i] = Vector_Transpose(self.functionsList[i].vector())*matrixOrVector
            return onlineVector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList_Transpose.__mul__.")

# Auxiliary class: multiplication of the transpose of a functions list with a matrix
class FunctionsList_Transpose__times__Matrix(object):
    def __init__(self, functionsList, matrix):
        assert isinstance(functionsList, FunctionsList)
        assert isinstance(matrix, TruthMatrix) or isinstance(matrix, OnlineMatrix_Type)
        self.functionsList = functionsList
        self.matrix = matrix
      
    # self * functionsList2 [used e.g. to compute Z^T*A*Z or S^T*X*S (return OnlineMatrix), or Riesz_A^T*X*Riesz_F (return OnlineVector)]
    def __mul__(self, functionsList2OrVector):
        assert (
            isinstance(functionsList2OrVector, FunctionsList)
                or
            isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                or
            isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction)
        )
        if isinstance(functionsList2OrVector, FunctionsList):
            assert len(self.functionsList) == len(functionsList2OrVector)
            dim = len(self.functionsList)
            onlineMatrix = OnlineMatrix(dim, dim)
            for j in range(dim):
                if isinstance(functionsList2OrVector[j], TruthFunction) or isinstance(functionsList2OrVector[j], OnlineFunction):
                    matrixTimesVectorj = self.matrix*functionsList2OrVector[j].vector()
                else:
                    assert isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                    matrixTimesVectorj = self.matrix*functionsList2OrVector[j]
                for i in range(dim):
                    onlineMatrix[i, j] = Vector_Transpose(self.functionsList[i])*matrixTimesVectorj
            return onlineMatrix
        elif (
            isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, OnlineVector_Type)
                or
            isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction)
        ):
            if isinstance(functionsList2OrVector, TruthVector) or isinstance(functionsList2OrVector, TruthFunction):
                assert isinstance(self.matrix, TruthMatrix)
            elif isinstance(functionsList2OrVector, OnlineVector_Type) or isinstance(functionsList2OrVector, OnlineFunction):
                assert isinstance(self.matrix, OnlineMatrix_Type)
            else: # impossible to arrive here anyway, thanks to the assert
                raise AssertionError("Invalid arguments in FunctionsList_Transpose__times__Matrix.__mul__.")
            if isinstance(functionsList2OrVector, TruthFunction) or isinstance(functionsList2OrVector, OnlineFunction):
                functionsList2OrVector = functionsList2OrVector.vector()
            dim = len(self.functionsList)
            onlineVector = OnlineVector(dim)
            matrixTimesVector = self.matrix*functionsList2OrVector
            for i in range(dim):
                onlineVector[i] = Vector_Transpose(self.functionsList[i])*matrixTimesVector
            return onlineVector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList_Transpose__times__Matrix.__mul__.")
     
#  @}
########################### end - OFFLINE STAGE - end ########################### 

