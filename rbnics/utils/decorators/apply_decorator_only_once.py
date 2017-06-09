# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.utils.decorators.extends import Extends
from rbnics.utils.decorators.override import override

def apply_decorator_only_once(Decorator):
    def decorator_with_only_one_application(Class):        
        if hasattr(Class, "only_once_decorators"):
            decorator_already_applied = Decorator in Class.only_once_decorators
        else:
            decorator_already_applied = False
    
        if not decorator_already_applied:
            DecoratedClass = Decorator(Class)
            DecoratedClass.only_once_decorators = list()
            if hasattr(Class, "only_once_decorators"):
                DecoratedClass.only_once_decorators.extend(Class.only_once_decorators)
            DecoratedClass.only_once_decorators.append(Decorator)
            return DecoratedClass
        else:
            return Class
    return decorator_with_only_one_application
