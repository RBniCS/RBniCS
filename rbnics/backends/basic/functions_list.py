# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from numbers import Number
from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import dict_of, list_of, overload, ThetaType, tuple_of
from rbnics.utils.mpi import parallel_io


def FunctionsList(backend, wrapping, online_backend, online_wrapping,
                  AdditionalIsFunction=None, ConvertAdditionalFunctionTypes=None):
    from rbnics.backends.common import TimeSeries  # cannot import at global scope due to cyclic dependence

    if AdditionalIsFunction is None:
        def _AdditionalIsFunction(arg):
            return False
        AdditionalIsFunction = _AdditionalIsFunction
    if ConvertAdditionalFunctionTypes is None:
        def _ConvertAdditionalFunctionTypes(arg):
            raise NotImplementedError("Please implement conversion of additional function types")
        ConvertAdditionalFunctionTypes = _ConvertAdditionalFunctionTypes

    class _FunctionsList(AbstractFunctionsList):
        def __init__(self, space, component):
            if component is None:
                self.space = space
            else:
                self.space = wrapping.get_function_subspace(space, component)
            self.mpi_comm = wrapping.get_mpi_comm(space)
            self._list = list()  # of functions
            self._precomputed_slices = Cache()  # from tuple to FunctionsList

        def enrich(self, functions, component=None, weights=None, copy=True):
            # Append to storage
            self._enrich(functions, component, weights, copy)
            # Reset precomputed slices
            self._precomputed_slices = Cache()
            # Prepare trivial precomputed slice
            self._precomputed_slices[0, len(self._list)] = self

        @overload(backend.Function.Type(), (None, str, dict_of(str, str)), (None, Number), bool)
        def _enrich(self, function, component, weight, copy):
            self._add_to_list(function, component, weight, copy)

        @overload((lambda cls: cls, list_of(backend.Function.Type()), tuple_of(backend.Function.Type())),
                  (None, str, dict_of(str, str)), (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            if weights is not None:
                assert len(weights) == len(functions)
                for (index, function) in enumerate(functions):
                    self._add_to_list(function, component, weights[index], copy)
            else:
                for function in functions:
                    self._add_to_list(function, component, None, copy)

        @overload(TimeSeries, (None, str, dict_of(str, str)), (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            self._enrich(functions._list, component, weights, copy)

        @overload(object, (None, str, dict_of(str, str)), (None, Number, list_of(Number)), bool)
        def _enrich(self, function, component, weight, copy):
            if AdditionalIsFunction(function):
                function = ConvertAdditionalFunctionTypes(function)
                assert weight is None or isinstance(weight, Number)
                self._add_to_list(function, component, weight, copy)
            elif isinstance(function, list):
                converted_function = list()
                for function_i in function:
                    if AdditionalIsFunction(function_i):
                        converted_function.append(ConvertAdditionalFunctionTypes(function_i))
                    else:
                        raise RuntimeError("Invalid function provided to FunctionsList.enrich()")
                assert weight is None or isinstance(weight, list)
                self._enrich(converted_function, component, weight, copy)
            else:
                raise RuntimeError("Invalid function provided to FunctionsList.enrich()")

        @overload(backend.Function.Type(), (None, str), (None, Number), bool)
        def _add_to_list(self, function, component, weight, copy):
            self._list.append(wrapping.function_extend_or_restrict(function, component, self.space, component,
                                                                   weight, copy))

        @overload(backend.Function.Type(), dict_of(str, str), (None, Number), bool)
        def _add_to_list(self, function, component, weight, copy):
            assert len(component) == 1
            for (component_from, component_to) in component.items():
                break
            self._list.append(wrapping.function_extend_or_restrict(function, component_from, self.space, component_to,
                                                                   weight, copy))

        def clear(self):
            self._list = list()
            # Reset precomputed slices
            self._precomputed_slices.clear()

        def save(self, directory, filename):
            self._save_Nmax(directory, filename)
            for (index, function) in enumerate(self._list):
                wrapping.function_save(function, directory, filename + "_" + str(index))

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
                function = backend.Function(self.space)
                wrapping.function_load(function, directory, filename + "_" + str(index))
                self.enrich(function)
            return True

        def _load_Nmax(self, directory, filename):
            def load_Nmax_task():
                with open(os.path.join(str(directory), filename + ".length"), "r") as length:
                    return int(length.readline())
            return parallel_io(load_Nmax_task, self.mpi_comm)

        @overload(online_backend.OnlineMatrix.Type(), )
        def __mul__(self, other):
            return wrapping.functions_list_mul_online_matrix(self, other, type(self))

        @overload((online_backend.OnlineVector.Type(), ThetaType), )
        def __mul__(self, other):
            return wrapping.functions_list_mul_online_vector(self, other)

        @overload(online_backend.OnlineFunction.Type(), )
        def __mul__(self, other):
            return wrapping.functions_list_mul_online_vector(self, online_wrapping.function_to_vector(other))

        def __len__(self):
            return len(self._list)

        @overload(int)
        def __getitem__(self, key):
            return self._list[key]

        @overload(slice)  # e.g. key = :N, return the first N functions
        def __getitem__(self, key):
            if key.start is not None:
                start = key.start
            else:
                start = 0
            assert key.step is None
            if key.stop is not None:
                stop = key.stop
            else:
                stop = len(self._list)

            assert start <= stop
            if start < stop:
                assert start >= 0
                assert start < len(self._list)
                assert stop > 0
                assert stop <= len(self._list)
            # elif start == stop
            #    trivial case which will result in an empty FunctionsList

            if (start, stop) not in self._precomputed_slices:
                output = _FunctionsList.__new__(type(self), self.space)
                output.__init__(self.space)
                if start < stop:
                    output._list = self._list[key]
                self._precomputed_slices[start, stop] = output
            return self._precomputed_slices[start, stop]

        @overload(int, backend.Function.Type())
        def __setitem__(self, key, item):
            self._list[key] = item

        @overload(int, object)
        def __setitem__(self, key, item):
            if AdditionalIsFunction(item):
                item = ConvertAdditionalFunctionTypes(item)
                self._list[key] = item
            else:
                raise RuntimeError("Invalid function provided to FunctionsList.__setitem__()")

        def __iter__(self):
            return self._list.__iter__()

    return _FunctionsList
