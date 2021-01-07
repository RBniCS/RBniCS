# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.backends.common.time_series import TimeSeries
from rbnics.utils.decorators import backend_for, list_of, overload


@backend_for("common", inputs=((list_of(Number), TimeSeries), (list_of(Number), TimeSeries)))
def assign(object_to, object_from):
    _assign(object_to, object_from)


@overload
def _assign(object_to: TimeSeries, object_from: TimeSeries):
    if object_from is not object_to:
        from rbnics.backends import assign
        assign(object_to._list, object_from._list)


@overload
def _assign(object_to: list_of(Number), object_from: list_of(Number)):
    if object_from is not object_to:
        del object_to[:]
        object_to.extend(object_from)
