# Copyright (C) 2015-2019 by the RBniCS authors
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
        self._time_step_size = (time_interval[1] - time_interval[0])/(len(function_over_time) - 1)
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
