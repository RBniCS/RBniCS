# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import empty as AffineExpansionSeparatedFormsStorageContent_Base


class AffineExpansionSeparatedFormsStorage(object):
    def __init__(self, Q):
        self._content = AffineExpansionSeparatedFormsStorageContent_Base((Q,), dtype=object)

    def __getitem__(self, key):
        return self._content[key]

    def __setitem__(self, key, item):
        self._content[key] = item

    def __len__(self):
        return self._content.size
