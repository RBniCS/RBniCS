# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import cache


@cache
def get_global_dof_to_local_dof_map(V, dofmap):
    local_to_global = dofmap.tabulate_local_to_global_dofs()
    local_size = dofmap.ownership_range()[1] - dofmap.ownership_range()[0]
    global_to_local = {global_: local for (local, global_) in enumerate(local_to_global) if local < local_size}
    return global_to_local
