# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import namedtuple


def OnlineSolveKwargsGenerator(**kwargs):
    OnlineSolveKwargsTuple = namedtuple("OnlineSolveKwargs", kwargs.keys())
    OnlineSolveKwargsTuple.__new__.__defaults__ = tuple(kwargs.values())

    def OnlineSolveKwargs(*args_, **kwargs_):
        return OnlineSolveKwargsTuple(*args_, **kwargs_)._asdict()

    return OnlineSolveKwargs
