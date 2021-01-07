# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import NonAffineExpansionStorage as AbstractNonAffineExpansionStorage


def NonAffineExpansionStorage(backend, wrapping):

    class _NonAffineExpansionStorage(AbstractNonAffineExpansionStorage):
        def __init__(self, content):
            self._content = tuple(backend.ParametrizedTensorFactory(op) for op in content)

        def __getitem__(self, key):
            return self._content[key]

        def __iter__(self):
            return iter(self._content)

        def __len__(self):
            return len(self._content)

    return _NonAffineExpansionStorage
