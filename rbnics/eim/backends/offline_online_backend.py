# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
