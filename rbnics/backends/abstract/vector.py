# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_online_backend


@abstract_online_backend
def Vector(N):
    pass


# Moreover, it should also expose a Type method containing the type of the returned instance
# This can be specified using the @backend_for decorator setting its output attribute
def _Vector_Type():
    return None


Vector.Type = _Vector_Type
