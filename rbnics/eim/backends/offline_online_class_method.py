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

from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.cache import cache

@cache
def OfflineOnlineClassMethod(problem_name):
    _OfflineOnlineClassMethod_Base = OfflineOnlineSwitch(problem_name)
    class _OfflineOnlineClassMethod(_OfflineOnlineClassMethod_Base):
        def __init__(self, problem, original_class_method_name):
            _OfflineOnlineClassMethod_Base.__init__(self)
            assert hasattr(problem, original_class_method_name)
            self._original_class_method = getattr(problem, original_class_method_name)
            self._replacement_condition = dict()
            
        def __call__(self, term):
            if self._replacement_condition[_OfflineOnlineClassMethod_Base._current_stage](term):
                return self._content[_OfflineOnlineClassMethod_Base._current_stage](term)
            else:
                return self._original_class_method(term)
            
        def attach(self, replaced_class_method, replacement_condition):
            if _OfflineOnlineClassMethod_Base._current_stage not in self._content:
                assert _OfflineOnlineClassMethod_Base._current_stage not in self._replacement_condition
                self._content[_OfflineOnlineClassMethod_Base._current_stage] = replaced_class_method
                self._replacement_condition[_OfflineOnlineClassMethod_Base._current_stage] = replacement_condition
            else:
                assert replaced_class_method == self._content[_OfflineOnlineClassMethod_Base._current_stage]
                # assert replacement_condition == self._replacement_condition[_OfflineOnlineClassMethod_Base._current_stage] # disabled because cannot easily compare lambda functions
                
    return _OfflineOnlineClassMethod
