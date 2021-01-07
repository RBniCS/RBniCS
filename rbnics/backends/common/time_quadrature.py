# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from scipy.integrate import simps
from rbnics.backends.abstract import TimeQuadrature as AbstractTimeQuadrature
from rbnics.backends.common.time_series import TimeSeries
from rbnics.utils.decorators import backend_for, list_of, tuple_of, overload


@backend_for("common", inputs=(tuple_of(Number), (list_of(Number), TimeSeries)))
def TimeQuadrature(time_interval, function_over_time):
    return _TimeQuadrature(time_interval, function_over_time)


class TimeQuadrature_Numbers(AbstractTimeQuadrature):
    def __init__(self, time_interval, function_over_time):
        assert len(function_over_time) > 1
        self._time_step_size = (time_interval[1] - time_interval[0]) / (len(function_over_time) - 1)
        self._function_over_time = function_over_time

    def integrate(self):
        return simps(self._function_over_time, dx=self._time_step_size)


@overload
def _TimeQuadrature(time_interval: tuple_of(Number), function_over_time: list_of(Number)):
    return TimeQuadrature_Numbers(time_interval, function_over_time)


@overload
def _TimeQuadrature(time_interval: tuple_of(Number), time_series: TimeSeries):
    from rbnics.backends import TimeQuadrature
    return TimeQuadrature(time_interval, time_series._list)
