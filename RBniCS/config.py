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
## @file config.py
#  @brief Configuration
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function

from dolfin import *

###########################     ONLINE STAGE     ########################### 
## @defgroup OnlineStage Methods related to the online stage
#  @{

# Declare reduced matrix type
from numpy import matrix as OnlineMatrix_Base
from numpy import zeros as OnlineMatrixContent_Base
def OnlineMatrix(M=None, N=None):
    assert (M is None and N is None) or (M is not None and N is not None)
    if M is not None and N is not None:
        return OnlineMatrix_Base(OnlineMatrixContent_Base((M, N)))
    else:
        return None
    
# Declare reduced vector type
from numpy import matrix as OnlineVector_Base
from numpy import zeros as OnlineVectorContent_Base
def OnlineVector(N=None):
    if N is not None:
        return OnlineVector_Base(OnlineVectorContent_Base((N))).transpose() # as column vector
    else:
        return None

# Declare reduced eigen solver type
from numpy import linalg.eig as OnlineEigenSolver_Impl
class OnlineEigenSolver(object):
    def __init__(self, A = None, B = None):
        if A is not None:
            assert A.shape[0] == A.shape[1]
        if B is not None:
            assert B.shape[0] == B.shape[1]
            assert A.shape[0] == B.shape[0]
        
        self.A = A
        self.B = B
        self.parameters = {}
        
    def solve(self):
        assert self.parameters["problem_type"] == "gen_hermitian" # only one implemented so far
        assert self.parameters["spectrum"] == "largest real" # only one implemented so far
        assert A is not None
        
        eigs, eigv = OnlineEigenSolver_Impl(self.A, self.B)
        
        idx = eigs.argsort()
        idx = idx[::-1]
        eigs = eigs[idx]
        eigv = eigv[:, idx]
        
        # Remove (negigible) complex parts
        eigs = np.real(eigs)
        eigv = np.real(eigv)
    
    def get_eigenvalue(self, i):
        return eigs[i]
        
    def get_eigenvector(self, i)
        return eigv[:, i]
        
    def save_eigenvalues_file(self, directory, filename):
        with open(directory + "/" + filename, "a") as outfile:
            for i in range(len(self.eigs)):
                file.write(str(i) + " " + str(self.eigs[i]))
    
    def save_retained_energy_file(self, directory, filename):
        from numpy import sum as np_sum
        from numpy import cumsum as np_cumsum
        energy = np_sum(self.eigs)
        eigs_cumsum = np_cumsum(self.eigs)
        eigs_cumsum /= energy
        with open(directory + "/" + filename, "a") as outfile:
            for i in range(len(eigs_cumsum)):
                file.write(str(i) + " " + str(eigs_cumsum[i])) 
    
#  @}
########################### end - ONLINE STAGE - end ########################### 

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Declare truth matrix type (from FEniCS)
TruthMatrix = PETScMatrix

# Declare truth vector type (from FEniCS)
TruthVector = PETScVector

# Declare truth eigen solver type (from FEniCS)
TruthEigenSolver = SLEPcEigenSolver
    
#  @}
########################### end - OFFLINE STAGE - end ########################### 

###########################     OFFLINE AND ONLINE COMMON INTERFACES     ########################### 
## @defgroup OfflineOnlineInterfaces Common interfaces for offline and online
#  @{

# Parameter space subsets
class ParameterSpaceSubset(object): # equivalent to a list of tuples
    def __init__(self):
        self._list = []
    
    # Method for generation of parameter space subsets
    # If the last argument is equal to "random", n parameters are drawn from a random uniform distribution
    # Else, if the last argument is equal to "linspace", (approximately) n parameters are obtained from a cartesian grid
    def generate(self, box, n, sampling):
        if sampling == "random":
            ss = "[("
            for i in range(len(box)):
                ss += "np.random.uniform(box[" + str(i) + "][0], box[" + str(i) + "][1])"
                if i < len(box)-1:
                    ss += ", "
                else:
                    ss += ") for _ in range(" + str(n) +")]"
            self._list = eval(ss)
        elif sampling == "linspace":
            n_P_root = int(np.ceil(n**(1./len(box))))
            ss = "itertools.product("
            for i in range(len(box)):
                ss += "np.linspace(box[" + str(i) + "][0], box[" + str(i) + "][1], num = " + str(n_P_root) + ").tolist()"
                if i < len(box)-1:
                    ss += ", "
                else:
                    ss += ")"
            self._list = eval(ss)
        else:
            raise RuntimeError("Invalid sampling mode.")

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
        
    def __getitem__(self, key):
        return self._list[key]

    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)

# Hide the implementation of an array with two or more indices, used to store tensors
# for error estimation. There are typically two kind of indices, either over the
# affine expansion or over the basis functions. This container will be indicized
# over the affine expansion. Its content will be another container, indicized over
# the basis functions
# Requires: access with operator[]
from numpy import empty as AffineExpansionStorageContent_Base
class AffineExpansionStorage(object):
    def __init__(self, *args):
        self._content = None
        if args:
            self._content = AffineExpansionStorageContent_Base(args, dtype=object)
            
    def load(self, directory, filename):
        if self._content: # avoid loading multiple times
            return False
        if io_utils.exists_numpy_file(directory, filename):
            self._content = io_utils.load_numpy_file(directory, filename)
            return True
        else:
            return False
        
    def save(self, directory, filename):
        io_utils.save_numpy_file(self._content, directory, filename)

    def __getitem__(self, key):
        return self._content[key]
        
    def __setitem__(self, key, item):
        self._content[key] = item
        
# Hide the implementation of a list of FE functions.
# Requires: access with operator[] and enrich
class FunctionsList(object):
    def __init__(self):
        self._list = []
    
    def enrich(functions):
        import collections
        if isinstance(functions, collections.Iterable): # more than one function
            self._list.extend(functions) # assume that they where already copied
        else: # one function
            self.list.append(functions.vector().copy()) # copy it explicitly
            
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
        
# Hide the implementation of the basis functions matrix.
# Requires: access with operator[] and enrich
BasisFunctionsMatrix = FunctionsList

# Hide the implementation of the snapshot matrix.
# Requires: access with operator[] and enrich
SnapshotsMatrix = FunctionsList

# Similarly to FEniCS' solve(A, x, b) define a solve for online problems
dolfin_solve = solve
def solve(A, x, b):
    if isinstance(A, TruthMatrix) and isinstance(x, TruthVector) and isinstance(b, TruthVector):
        dolfin_solve(A, x, b)
    elif isinstance(A, OnlineMatrix) and isinstance(x, OnlineVector) and isinstance(b, OnlineVector):
        x = np.linalg.solve(A, b)
    else:
        raise RuntimeError("Invalid input arguments in solve")

class utils(object):
            
    ## Auxiliary internal methods to compute scalar products (v1, v2) or (v1, M*v2)
    @staticmethod
    def compute_scalar_product(arg1, arg2, arg3=None):
        if arg3 is None:
        
            ## Auxiliary internal methods to compute scalar product (v1, v2)
            def _compute_scalar_product__v1v2(v1, v2):
                assert \
                    (isinstance(v1, GenericVector) and isinstance(v2, GenericVector)) \
                        or \
                    (isinstance(v1, Function) and isinstance(v2, Function))
                #
                if isinstance(v1, GenericVector) and isinstance(v2, GenericVector):
                    return v1.inner(M*v2)
                elif isinstance(v1, Function) and isinstance(v2, Function):
                    return v1.vector().inner(M*v2.vector())
                else: # impossible to arrive here anyway, thanks to the assert
                    raise RuntimeError("Invalid arguments in compute_scalar_product.")
            
            _compute_scalar_product__v1v2(arg1, arg2)
        
        else:
        
            ## Auxiliary internal methods to compute scalar product (v1, M*v2)
            def _compute_scalar_product__v1Mv2(v1, M, v2):
                assert isinstance(M, GenericMatrix)
                assert \
                    (isinstance(v1, GenericVector) and isinstance(v2, GenericVector)) \
                        or \
                    (isinstance(v1, Function) and isinstance(v2, Function))
                #
                if isinstance(v1, GenericVector) and isinstance(v2, GenericVector):
                    return v1.inner(M*v2)
                elif isinstance(v1, Function) and isinstance(v2, Function):
                    return v1.vector().inner(M*v2.vector())
                else: # impossible to arrive here anyway, thanks to the assert
                    raise RuntimeError("Invalid arguments in compute_scalar_product.")
            
            _compute_scalar_product__v1Mv2(arg1, arg2, arg3)
        
#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 

###########################     ERROR ANALYSIS     ########################### 
## @defgroup ErrorAnalysis Error analysis
#  @{

from numpy import log, exp, mean, sqrt

#  @}
########################### end - ERROR ANALYSIS - end ########################### 

###########################     I/O     ########################### 
## @defgroup IO Input/output methods
#  @{

# Override the print() method to print only from process 0 in parallel
import __builtin__

def print(*args, **kwargs):
    if MPI.rank(print.mpi_comm) == 0:
        return __builtin__.print(*args, **kwargs)

print.mpi_comm = mpi_comm_world() # from dolfin

class io_utils(object):
    
    ## Load a variable from file using pickle
    @staticmethod
    def load_pickle_file(directory, filename):
        with open(directory + "/" + filename + ".pkl", "rb") as infile:
            return pickle.load(infile)
    
    ## Save a variable to file using pickle
    @staticmethod
    def save_pickle_file(subset, directory, filename):
        with open(directory + "/" + filename + ".pkl", "wb") as outfile:
            pickle.dump(subset, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            
    ## Check if a pickle file exists
    @staticmethod
    def exists_pickle_file(directory, filename):
        return os.path.exists(directory + "/" + filename + ".pkl")
        
    ## Load a variable from file using numpy
    @staticmethod
    def load_numpy_file(directory, filename):
        return numpy.load(directory + "/" + filename + ".npy")
    
    ## Save a variable to file using numpy
    @staticmethod
    def save_numpy_file(subset, directory, filename):
        np.save(directory + "/" + filename, subset)
            
    ## Check if a numpy file exists
    @staticmethod
    def exists_numpy_file(directory, filename):
        return os.path.exists(directory + "/" + filename + ".npy")

#  @}
########################### end - I/O - end ########################### 
    
