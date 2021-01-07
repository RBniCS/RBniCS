# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.utils.decorators.dispatch import dict_of, tuple_of


def ComputeThetaType(additional_types=None):
    all_types = [Number]
    if additional_types is not None:
        all_types.extend(additional_types)
    all_types = tuple(all_types)
    return (tuple_of(all_types), )


ThetaType = ComputeThetaType()
DictOfThetaType = (dict_of(str, ThetaType), )
