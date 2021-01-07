# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# assign function_from to function_to storage
@abstract_backend
def assign(object_to, object_from):
    pass
