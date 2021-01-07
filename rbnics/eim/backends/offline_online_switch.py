# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.cache import cache


@cache
def OfflineOnlineSwitch(problem_name):

    class _OfflineOnlineSwitch(object):
        _current_stage = "offline"

        def __init__(self):
            self._content = dict()

        @classmethod
        def set_current_stage(cls, current_stage):
            assert current_stage in ("offline", "online")
            cls._current_stage = current_stage

        @classmethod
        def get_current_stage(cls):
            return cls._current_stage

    return _OfflineOnlineSwitch
