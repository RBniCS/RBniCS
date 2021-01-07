# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.cache import cache


@cache
def OfflineOnlineExpansionStorageSize(problem_name):

    _OfflineOnlineExpansionStorageSize_Base = OfflineOnlineSwitch(problem_name)

    class _OfflineOnlineExpansionStorageSize(_OfflineOnlineExpansionStorageSize_Base):
        def __init__(self):
            _OfflineOnlineExpansionStorageSize_Base.__init__(self)
            self._content = {
                "offline": dict(),
                "online": dict()
            }

        def __getitem__(self, term):
            return self._content[_OfflineOnlineExpansionStorageSize_Base._current_stage][term]

        def __setitem__(self, term, size):
            self._content[_OfflineOnlineExpansionStorageSize_Base._current_stage][term] = size

        def __contains__(self, term):
            return term in self._content[_OfflineOnlineExpansionStorageSize_Base._current_stage]

    return _OfflineOnlineExpansionStorageSize
