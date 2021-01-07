# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import backend_for, OnlineSizeType
from rbnics.backends.online.basic import Function as BasicFunction
from rbnics.backends.online.numpy.vector import Vector

_Function_Type = BasicFunction(Vector)


@backend_for("numpy", inputs=(OnlineSizeType + (Vector.Type(), ), ))
def Function(arg):
    return _Function_Type(arg)


# Attach a Type() function
def Type():
    return _Function_Type


Function.Type = Type
