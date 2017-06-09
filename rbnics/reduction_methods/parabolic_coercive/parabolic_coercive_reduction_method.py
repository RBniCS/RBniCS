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

from rbnics.reduction_methods.base import TimeDependentReductionMethod
from rbnics.problems.parabolic_coercive.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.utils.decorators import Extends, override

# Base class containing the interface of a projection based ROM
# for parabolic coercive problems.
def ParabolicCoerciveReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    
    ParabolicCoerciveReductionMethod_Base = TimeDependentReductionMethod(EllipticCoerciveReductionMethod_DerivedClass)
    
    @Extends(ParabolicCoerciveReductionMethod_Base)
    class ParabolicCoerciveReductionMethod_Class(ParabolicCoerciveReductionMethod_Base):
        pass
        
    # return value (a class) for the decorator
    return ParabolicCoerciveReductionMethod_Class
        
