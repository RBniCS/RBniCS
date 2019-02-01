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
from numpy import arange, isclose
from rbnics.backends.abstract import TimeSeries as AbstractTimeSeries
from rbnics.utils.decorators import BackendFor, overload, tuple_of

@BackendFor("common", inputs=((tuple_of(Number), AbstractTimeSeries), (Number, None)))
class TimeSeries(AbstractTimeSeries):
    def __init__(self, *args):
        assert len(args) in (1, 2)
        if len(args) == 1:
            other_time_series, = args
            assert isinstance(other_time_series, TimeSeries)
            self._time_interval = other_time_series._time_interval
            self._time_step_size = other_time_series._time_step_size
        else:
            time_interval, time_step_size = args
            self._time_interval = time_interval
            self._time_step_size = time_step_size
        self._times = arange(self._time_interval[0], self._time_interval[1] + self._time_step_size/2., self._time_step_size).tolist()
        self._list = list()
        
    def stored_times(self):
        return self._times[:len(self._list)]
        
    def expected_times(self):
        return self._times
        
    @overload(int)
    def __getitem__(self, key):
        return self._list[key]
        
    @overload(slice)
    def __getitem__(self, key):
        if key.start is None:
            time_interval_0 = self._time_interval[0]
        else:
            time_interval_0 = self._time_interval[0] + key.start*self._time_step_size
        if key.step is None:
            time_step_size = self._time_step_size
        else:
            time_step_size = key.step*self._time_step_size
        if key.stop is None:
            time_interval_1 = self._time_interval[1]
        else:
            time_interval_1 = self._time_interval[0] + (key.stop - 1)*self._time_step_size
        output = TimeSeries((time_interval_0, time_interval_1), time_step_size)
        output.extend(self._list[key])
        return output
        
    def at(self, time):
        assert time >= self._time_interval[0]
        assert time <= self._time_interval[1]
        index = int(round(time/self._time_step_size))
        assert isclose(index*self._time_step_size, time), "Requested time should be a multiple of discretization time step size"
        return self._list[index]
        
    def __iter__(self):
        return iter(self._list)
        
    def __len__(self):
        return len(self._list)
        
    def __delitem__(self, key):
        del self._list[key]
        
    def append(self, item):
        self._list.append(item)
        
    def extend(self, iterable):
        self._list.extend(iterable)
        
    def clear(self):
        del self._list[:]
        
    def __str__(self):
        return str([e if isinstance(e, Number) else str(e) for e in self._list])
