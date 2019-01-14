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

import types
from rbnics.eim.backends.offline_online_class_method import OfflineOnlineClassMethod
from rbnics.eim.backends.offline_online_expansion_storage import OfflineOnlineExpansionStorage
from rbnics.eim.backends.offline_online_expansion_storage_size import OfflineOnlineExpansionStorageSize
from rbnics.eim.backends.offline_online_riesz_solver import OfflineOnlineRieszSolver
from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.cache import cache

@cache
def OfflineOnlineBackend(problem_name):
    return types.SimpleNamespace(
        OfflineOnlineClassMethod=OfflineOnlineClassMethod(problem_name),
        OfflineOnlineExpansionStorage=OfflineOnlineExpansionStorage(problem_name),
        OfflineOnlineExpansionStorageSize=OfflineOnlineExpansionStorageSize(problem_name),
        OfflineOnlineRieszSolver=OfflineOnlineRieszSolver(problem_name),
        OfflineOnlineSwitch=OfflineOnlineSwitch(problem_name)
    )
