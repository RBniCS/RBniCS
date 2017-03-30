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
## @file numpy_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import inspect

def Extends(Parent_, preserve_class_name=False):
    def Extends_Decorator(Child):
        
        assert inspect.isclass(Parent_) or isinstance(Parent_, tuple)
        if inspect.isclass(Parent_):
            Parents = (Parent_, )
        else:
            Parents = Parent_
        
        Bases = Child.__bases__
        assert len(Bases) == len(Parents), "Child class has more than base classes than the ones provided to Extends"
        for (Parent, Base) in zip(Parents, Bases):
            Parent_is_Base = False
            if Base is Parent:
                Parent_is_Base = True
            else:
                Ancestors = inspect.getmro(Base)
                for Ancestor in Ancestors:
                    if Ancestor is Parent:
                        Parent_is_Base = True
                        break
            assert Parent_is_Base, "The class " + str(Parent) + ", that you have indicated as Parent, is not a base class"
        
        # TODO save the class documentation from the parent class
        
        if preserve_class_name:
            assert len(Parents) == 1
            Parent = Parents[0]
            setattr(Child, "__name__", Parent.__name__)
            setattr(Child, "__module__", Parent.__module__)
        
        return Child
    return Extends_Decorator

