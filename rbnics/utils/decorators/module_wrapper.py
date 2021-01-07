# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from types import SimpleNamespace


def ModuleWrapper(*args, **kwargs):
    for arg in args:
        assert arg.__name__ not in kwargs
        kwargs[arg.__name__] = arg
    return SimpleNamespace(**kwargs)
