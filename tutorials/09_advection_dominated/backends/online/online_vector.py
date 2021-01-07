# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online import OnlineVector as _OnlineVector


def OnlineVector(N):
    assert isinstance(N, int)
    return _OnlineVector({"u": N})
