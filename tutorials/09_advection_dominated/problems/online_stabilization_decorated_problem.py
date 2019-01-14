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

from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor

def OnlineStabilizationDecoratedProblem(**decorator_kwargs):
    from .online_stabilization import OnlineStabilization
    
    @ProblemDecoratorFor(OnlineStabilization)
    def OnlineStabilizationDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):
        
        @PreserveClassName
        class OnlineStabilizationDecoratedProblem_Class(EllipticCoerciveProblem_DerivedClass):
            
            def __init__(self, V, **kwargs):
                # Flag to enable or disable stabilization
                self.stabilized = True
                # Call to parent
                EllipticCoerciveProblem_DerivedClass.__init__(self, V, **kwargs)
                        
        # return value (a class) for the decorator
        return OnlineStabilizationDecoratedProblem_Class
    
    # return the decorator itself
    return OnlineStabilizationDecoratedProblem_Decorator
