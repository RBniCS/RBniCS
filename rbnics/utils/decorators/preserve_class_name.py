# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def PreserveClassName(Derived):
    Parents = Derived.__bases__
    assert len(Parents) == 1
    Parent = Parents[0]
    setattr(Derived, "__name__", Parent.__name__)
    setattr(Derived, "__module__", Parent.__module__)
    return Derived
