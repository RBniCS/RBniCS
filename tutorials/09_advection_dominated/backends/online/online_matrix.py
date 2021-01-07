# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online import OnlineMatrix as _OnlineMatrix


def OnlineMatrix(M, N):
    assert isinstance(M, int)
    assert isinstance(N, int)
    return _OnlineMatrix({"u": M}, {"u": N})
