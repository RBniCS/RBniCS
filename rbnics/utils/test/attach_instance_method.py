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

class AttachInstanceMethod(object):
    def __init__(self, instance, method_name, patched_method):
        self._instance = instance
        self._method_name = method_name
        self._patched_method = patched_method
        assert not hasattr(instance, method_name)
        
    def attach(self):
        setattr(self._instance, self._method_name, types.MethodType(self._patched_method, self._instance))
