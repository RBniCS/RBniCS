# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch

def OfflineOnlineExpansionStorageSize(problem_name):
    if problem_name not in _offline_online_expansion_storage_size_cache:
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
        
        _offline_online_expansion_storage_size_cache[problem_name] = _OfflineOnlineExpansionStorageSize
        
    return _offline_online_expansion_storage_size_cache[problem_name]
        
_offline_online_expansion_storage_size_cache = dict()
