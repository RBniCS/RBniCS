# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from rbnics.utils.cache import cache

@cache
def get_global_dof_to_local_dof_map(V, dofmap):
    local_to_global = dofmap.tabulate_local_to_global_dofs()
    local_size = dofmap.ownership_range()[1] - dofmap.ownership_range()[0]
    global_to_local = {global_: local for (local, global_) in enumerate(local_to_global) if local < local_size}
    return global_to_local
