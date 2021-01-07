# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


@abstract_backend
def Function(V_or_N, component=None):
    pass


# Moreover, it should also expose a Type method containing the type of the returned instance.
# This can be specified using the @backend_for decorator setting its output attribute
def _FunctionType():
    return None


Function.Type = _FunctionType
