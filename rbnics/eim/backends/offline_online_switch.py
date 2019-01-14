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
