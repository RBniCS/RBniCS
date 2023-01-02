# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_online_backend


@abstract_online_backend
def Matrix(M, N):
    pass


# Moreover, it should also expose a Type method containing the type of the returned instance
# This can be specified using the @backend_for decorator setting its output attribute
def _Matrix_Type():
    return None


Matrix.Type = _Matrix_Type
