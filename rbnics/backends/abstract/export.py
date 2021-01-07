# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import abstract_backend


# Export a solution to file
@abstract_backend
def export(solution, directory, filename, suffix=None, component=None):
    pass
