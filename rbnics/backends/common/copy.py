# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.common.time_series import TimeSeries
from rbnics.utils.decorators import backend_for


@backend_for("common", inputs=(TimeSeries, ))
def copy(time_series):
    from rbnics.backends import copy
    time_series_copy = TimeSeries(time_series._time_interval, time_series._time_step_size)
    if len(time_series._list) > 0:
        time_series_copy._list = copy(time_series._list)
    return time_series_copy
