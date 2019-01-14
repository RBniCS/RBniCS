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

from rbnics.problems.parabolic.abstract_parabolic_reduced_problem import AbstractParabolicReducedProblem

# Base class containing the interface of a projection based ROM
# for parabolic problems.
def ParabolicReducedProblem(EllipticReducedProblem_DerivedClass):
    
    ParabolicReducedProblem_Base = AbstractParabolicReducedProblem(EllipticReducedProblem_DerivedClass)
    
    class ParabolicReducedProblem_Class(ParabolicReducedProblem_Base):
        pass
            
    # return value (a class) for the decorator
    return ParabolicReducedProblem_Class
