# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import overload
from rbnics.utils.mpi import parallel_io


def TensorsList(backend, wrapping, online_backend, online_wrapping):

    class _TensorsList(AbstractTensorsList):
        def __init__(self, space, empty_tensor):
            self.space = space
            self.empty_tensor = empty_tensor
            self.mpi_comm = wrapping.get_mpi_comm(space)
            self._list = list()  # of tensors
            self._precomputed_slices = Cache()  # from tuple to TensorsList

        def enrich(self, tensors):
            # Append to storage
            self._enrich(tensors)
            # Reset precomputed slices
            self._precomputed_slices.clear()
            # Prepare trivial precomputed slice
            self._precomputed_slices[0, len(self._list)] = self

        @overload((backend.Matrix.Type(), backend.Vector.Type()), )
        def _enrich(self, tensors):
            self._list.append(wrapping.tensor_copy(tensors))

        @overload(lambda cls: cls, )
        def _enrich(self, tensors):
            for tensor in tensors:
                self._list.append(wrapping.tensor_copy(tensor))

        def clear(self):
            self._list = list()
            # Reset precomputed slices
            self._precomputed_slices.clear()

        def save(self, directory, filename):
            self._save_Nmax(directory, filename)
            for (index, tensor) in enumerate(self._list):
                wrapping.tensor_save(tensor, directory, filename + "_" + str(index))

        def _save_Nmax(self, directory, filename):
            def save_Nmax_task():
                with open(os.path.join(str(directory), filename + ".length"), "w") as length:
                    length.write(str(len(self._list)))
            parallel_io(save_Nmax_task, self.mpi_comm)

        def load(self, directory, filename):
            if len(self._list) > 0:  # avoid loading multiple times
                return False
            Nmax = self._load_Nmax(directory, filename)
            for index in range(Nmax):
                tensor = wrapping.tensor_copy(self.empty_tensor)
                wrapping.tensor_load(tensor, directory, filename + "_" + str(index))
                self.enrich(tensor)
            return True

        def _load_Nmax(self, directory, filename):
            def load_Nmax_task():
                with open(os.path.join(str(directory), filename + ".length"), "r") as length:
                    return int(length.readline())
            return parallel_io(load_Nmax_task, self.mpi_comm)

        @overload(online_backend.OnlineFunction.Type(), )
        def __mul__(self, other):
            return wrapping.tensors_list_mul_online_function(self, other)

        def __len__(self):
            return len(self._list)

        @overload(int)
        def __getitem__(self, key):
            return self._list[key]

        @overload(slice)  # e.g. key = :N, return the first N tensors
        def __getitem__(self, key):
            if key.start is not None:
                start = key.start
                assert start >= 0
                assert start < len(self._list)
            else:
                start = 0
            assert key.step is None
            if key.stop is not None:
                stop = key.stop
                assert stop > 0
                assert stop <= len(self._list)
            else:
                stop = len(self._list)

            if (start, stop) not in self._precomputed_slices:
                output = _TensorsList.__new__(type(self), self.space, self.empty_tensor)
                output.__init__(self.space, self.empty_tensor)
                output._list = self._list[key]
                self._precomputed_slices[start, stop] = output
            return self._precomputed_slices[start, stop]

        def __iter__(self):
            return self._list.__iter__()

    return _TensorsList
