# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from scipy.integrate import simps
from rbnics.backends.abstract import TimeQuadrature as AbstractTimeQuadrature
from rbnics.backends.online.numpy.function import Function
from rbnics.utils.decorators import BackendFor, list_of, tuple_of


@BackendFor("numpy", inputs=(tuple_of(Number), list_of(Function.Type())))
class TimeQuadrature(AbstractTimeQuadrature):
    def __init__(self, time_interval, function_over_time):
        assert len(function_over_time) > 1
        self._time_step_size = (time_interval[1] - time_interval[0]) / (len(function_over_time) - 1)
        self._function_over_time = function_over_time

    def integrate(self):
        vector_over_time = list()
        N = self._function_over_time[0].N
        for function in self._function_over_time:
            assert function.N == N
            vector_over_time.append(function.vector())
        integrated_vector = simps(vector_over_time, dx=self._time_step_size, axis=0)
        integrated_function = Function(N)
        integrated_function.vector()[:] = integrated_vector
        return integrated_function
