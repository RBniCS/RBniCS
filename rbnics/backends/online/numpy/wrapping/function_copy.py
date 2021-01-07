# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic.wrapping.function_copy import basic_function_copy  # noqa: F401

# No explicit instantiation for backend = rbnics.backends.online.numpy to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.online.numpy.copy
