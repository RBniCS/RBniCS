# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import ExportableList


class BoundingBoxSideList(ExportableList):
    def __init__(self, size=None):
        ExportableList.__init__(self, "text")
        if size is not None:
            self._list.extend([0. for _ in range(size)])
