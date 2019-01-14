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

from rbnics.backends.common.time_series import TimeSeries
from rbnics.utils.decorators import backend_for

@backend_for("common", inputs=(TimeSeries, ))
def copy(time_series):
    from rbnics.backends import copy
    time_series_copy = TimeSeries(time_series._time_interval, time_series._time_step_size)
    if len(time_series._list) > 0:
        time_series_copy._list = copy(time_series._list)
    return time_series_copy
