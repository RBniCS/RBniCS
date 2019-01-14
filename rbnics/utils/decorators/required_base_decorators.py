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

def RequiredBaseDecorators(*BaseDecorators):
    def RequiredBaseDecorators_FunctionDecorator(Decorator):
        def RequiredBaseDecorators_ClassDecorator(Class):
            BaseClass = Class
            AlreadyAppliedBaseDecorators = list()
            if hasattr(Class, "AlreadyAppliedBaseDecorators"):
                AlreadyAppliedBaseDecorators.extend(Class.AlreadyAppliedBaseDecorators)
            
            for BaseDecorator in BaseDecorators:
                if (
                    BaseDecorator is not None
                        and
                    BaseDecorator.__name__ not in AlreadyAppliedBaseDecorators
                ):
                    BaseClass = BaseDecorator(BaseClass)
                    AlreadyAppliedBaseDecorators.append(BaseDecorator.__name__)
            
            if Decorator not in AlreadyAppliedBaseDecorators:
                DecoratedClass = Decorator(BaseClass)
                DecoratedClass.AlreadyAppliedBaseDecorators = list()
                DecoratedClass.AlreadyAppliedBaseDecorators.extend(AlreadyAppliedBaseDecorators)
                DecoratedClass.AlreadyAppliedBaseDecorators.append(Decorator.__name__)
                return DecoratedClass
            else:
                return BaseClass
        
        # Preserve class decorator name and return
        setattr(RequiredBaseDecorators_ClassDecorator, "__name__", Decorator.__name__)
        setattr(RequiredBaseDecorators_ClassDecorator, "__module__", Decorator.__module__)
        return RequiredBaseDecorators_ClassDecorator
        
    return RequiredBaseDecorators_FunctionDecorator
