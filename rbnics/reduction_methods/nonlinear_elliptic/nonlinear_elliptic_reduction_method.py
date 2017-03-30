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

from rbnics.reduction_methods.base import NonlinearReductionMethod
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from rbnics.utils.decorators import Extends, override, MultiLevelReductionMethod

def NonlinearEllipticReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    @Extends(EllipticCoerciveReductionMethod_DerivedClass) # needs to be first in order to override for last the methods.
    #@ReductionMethodFor(NonlinearEllipticProblem, "Abstract") # disabled, since now this is a decorator which depends on a derived (e.g. POD or RB) class
    @MultiLevelReductionMethod
    @NonlinearReductionMethod
    class NonlinearEllipticReductionMethod_Class(EllipticCoerciveReductionMethod_DerivedClass):
        
        ## Default initialization of members
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
        
    # return value (a class) for the decorator
    return NonlinearEllipticReductionMethod_Class
        
