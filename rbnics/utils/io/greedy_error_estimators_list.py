# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io.exportable_list import ExportableList


class GreedyErrorEstimatorsList(ExportableList):
    def __init__(self):
        ExportableList.__init__(self, "text")
