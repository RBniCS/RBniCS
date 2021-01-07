# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
                # assert replacement_condition == self._replacement_condition[
                #    _OfflineOnlineClassMethod_Base._current_stage]
                # assert above has been disabled because we cannot easily compare lambda functions

    return _OfflineOnlineClassMethod
