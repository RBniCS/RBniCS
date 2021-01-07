# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators import backend_for
python_abs = abs


@backend_for("common", inputs=(Number, ))
def abs(arg):
    return python_abs(arg)
