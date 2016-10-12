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
## @file affine_expansion_online_storage.py
#  @brief Type for storing online quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import empty as AffineExpansionStorageContent_Base
from numpy import nditer as AffineExpansionStorageContent_Iterator
from numpy import asmatrix as AffineExpansionStorageContent_AsMatrix
from numpy import ix_ as AffineExpansionStorageContent_Slicer
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage, FunctionsList as AbstractFunctionsList
from RBniCS.backends.numpy.matrix import Matrix as OnlineMatrix
from RBniCS.backends.numpy.vector import Vector as OnlineVector
from RBniCS.utils.io import NumpyIO as AffineExpansionStorageContent_IO
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractAffineExpansionStorage)
@BackendFor("NumPy", inputs=((int, AbstractAffineExpansionStorage), (int, None)))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, arg1, arg2=None):
        self._content = None
        self._content_as_matrix = None
        self._precomputed_slices = dict() # from tuple to AffineExpansionStorage
        self._recursive = False
        # Carry out initialization
        assert (
            (isinstance(arg1, int) and (isinstance(arg2, int) or arg2 is None))
                or
            (isinstance(arg1, AbstractAffineExpansionStorage) and arg2 is None)
        )
        if isinstance(arg1, AbstractAffineExpansionStorage):
            self._recursive = True
            self._content = arg1._content
            self._content_as_matrix = arg1._content_as_matrix
            self._precomputed_slices = arg1._precomputed_slices
        elif isinstance(arg1, int):
            if arg2 is None:
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, ), dtype=object)
            else:
                assert isinstance(arg2, int)
                self._recursive = False
                self._content = AffineExpansionStorageContent_Base((arg1, arg2), dtype=object)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to AffineExpansionStorage")
        # Auxiliary storage for __getitem__ slicing
        self._component_name_to_basis_component_index = None # will be filled in in __setitem__, if required
        self._component_name_to_basis_component_length = None # will be filled in in __setitem__, if required
    
    @override
    def load(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        if self._content is not None: # avoid loading multiple times
            if self._content.size > 0:
                it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                while not it.finished:
                    if self._content[it.multi_index] is not None: # ... but only if there is at least one element different from None
                        return False
                    it.iternext()
        if AffineExpansionStorageContent_IO.exists_file(directory, filename):
            self._content = AffineExpansionStorageContent_IO.load_file(directory, filename)
            # Create internal copy as matrix
            self._content_as_matrix = None
            self.as_matrix()
            # Reset precomputed slices
            it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
            self._prepare_trivial_precomputed_slice(self._content[it.multi_index])
            return True
        else:
            return False
            
    def _prepare_trivial_precomputed_slice(self, item):
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        if isinstance(item, OnlineMatrix.Type()):
            slice_0 = tuple(range(item.shape[0]))
            slice_1 = tuple(range(item.shape[1]))
            self._precomputed_slices[(slice_0, slice_1)] = self
        elif isinstance(item, OnlineVector.Type()):
            slice_0 = tuple(range(item.shape[0]))
            self._precomputed_slices[(slice_0, )] = self
                
    @override
    def save(self, directory, filename):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        AffineExpansionStorageContent_IO.save_file(self._content, directory, filename)
        
    def as_matrix(self):
        if self._content_as_matrix is None:
            self._content_as_matrix = AffineExpansionStorageContent_AsMatrix(self._content)
        return self._content_as_matrix
    
    @override
    def __getitem__(self, key):
        if (
            isinstance(key, slice)
                or
            ( isinstance(key, tuple) and isinstance(key[0], slice) )
        ): # return the subtensors of size "key" for every element in content. (e.g. submatrices [1:5,1:5] of the affine expansion of A)
            
            if isinstance(key, slice):
                key = (key,)
                
            assert isinstance(key, tuple)
            assert isinstance(key[0], slice)
            
            slices_start = list()
            slices_stop = list()
            for slice_ in key:
                assert slice_.start is None 
                assert slice_.step is None
                assert isinstance(slice_.stop, (int, dict))
                if isinstance(slice_.stop, int):
                    slices_start.append(0)
                    slices_stop.append(slice_.stop)
                else:
                    assert self._component_name_to_basis_component_index is not None
                    assert self._component_name_to_basis_component_length is not None
                    current_slice_start = [0]*len(self._component_name_to_basis_component_index)
                    current_slice_stop  = [0]*len(self._component_name_to_basis_component_index)
                    for (component_name, basis_component_index) in self._component_name_to_basis_component_index.iteritems():
                        current_slice_start[basis_component_index] = self._component_name_to_basis_component_length[component_name]
                        current_slice_stop[basis_component_index]  = current_slice_start[basis_component_index] + slice_.stop[component_name]
                    slices_start.append(current_slice_start)
                    slices_stop .append(current_slice_stop )
                    
            slices = list()
            assert len(slices_start) == len(slices_stop)
            for (current_slice_start, current_slice_stop) in zip(slices_start, slices_stop):
                assert isinstance(current_slice_start, int) == isinstance(current_slice_stop, int)
                if isinstance(current_slice_start, int):
                    slices.append(tuple(range(current_slice_start, current_slice_stop)))
                else:
                    current_slice = list()
                    for (current_slice_start_component, current_slice_stop_component) in zip(current_slice_start, current_slice_stop):
                        current_slice.extend(range(current_slice_start_component, current_slice_stop_component))
                    slices.append(tuple(current_slice))
            slices = tuple(slices)
            
            if slices in self._precomputed_slices:
                return self._precomputed_slices[slices]
            else:
                slices_type = AffineExpansionStorageContent_Slicer(*slices)
                output = AffineExpansionStorage(*self._content.shape)
                it = AffineExpansionStorageContent_Iterator(self._content, flags=["multi_index", "refs_ok"], op_flags=["readonly"])
                while not it.finished:
                    item = self._content[it.multi_index]
                    assert isinstance(item, (OnlineMatrix.Type(), OnlineVector.Type()))
                    output[it.multi_index] = item[slices_type]
                    it.iternext()
                self._precomputed_slices[slices] = output
                return output
                
        else: # return the element at position "key" in the storage (e.g. q-th matrix in the affine expansion of A, q = 1 ... Qa)
            return self._content[key]
        
    @override
    def __setitem__(self, key, item):
        assert not self._recursive # this method is used when employing this class online, while the recursive one is used offline
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._content[key] = item
        # Also store component_name_to_basis_component_* for __getitem__ slicing
        assert isinstance(item, (
            OnlineMatrix.Type(),  # output e.g. of Z^T*A*Z
            OnlineVector.Type(),  # output e.g. of Z^T*F
            float,                # output of Riesz_F^T*X*Riesz_F
            AbstractFunctionsList # auxiliary storage of Riesz representors
        ))
        assert hasattr(item, "_component_name_to_basis_component_index") == hasattr(item, "_component_name_to_basis_component_length")
        if hasattr(item, "_component_name_to_basis_component_index"): # temporarily added by transpose() method
            assert isinstance(item, (OnlineMatrix.Type(), OnlineVector.Type()))
            assert (self._component_name_to_basis_component_index is None) == (self._component_name_to_basis_component_length is None)
            if self._component_name_to_basis_component_index is None:
                self._component_name_to_basis_component_index = item._component_name_to_basis_component_index
                self._component_name_to_basis_component_length = item._component_name_to_basis_component_length
            else:
                assert self._component_name_to_basis_component_index == item._component_name_to_basis_component_index
                assert self._component_name_to_basis_component_length == item._component_name_to_basis_component_length
            del item._component_name_to_basis_component_index # cleanup temporary addition
            del item._component_name_to_basis_component_length # cleanup temporary addition
        else:
            assert self._component_name_to_basis_component_index is None
            assert self._component_name_to_basis_component_length is None
        # Reset internal copies
        self._content_as_matrix = None
        # Reset and prepare precomputed slices
        self._prepare_trivial_precomputed_slice(item)
        
    @override
    def __iter__(self):
        return AffineExpansionStorageContent_Iterator(self._content, flags=["refs_ok"], op_flags=["readonly"])
        
    @override
    def __len__(self):
        assert self.order() == 1
        return self._content.size
    
    def order(self):
        assert self._content is not None
        return len(self._content.shape)
        
