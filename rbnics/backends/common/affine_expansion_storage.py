# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from rbnics.utils.decorators import BackendFor, tuple_of


@BackendFor("common", inputs=(tuple_of(Number),))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    def __init__(self, args):
        self._content = args

    def __getitem__(self, key):
        return self._content[key]

    def __iter__(self):
        return self._content.__iter__()

    def __len__(self):
        assert self._content is not None
        return len(self._content)
