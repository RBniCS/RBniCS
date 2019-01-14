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

def OnlineVanishingViscosityDecoratedProblem(viscosity, N_threshold_min, N_threshold_max, **decorator_kwargs):
    from .online_vanishing_viscosity import OnlineVanishingViscosity
    
    @ProblemDecoratorFor(
        OnlineVanishingViscosity,
        viscosity=viscosity,
        N_threshold_min=N_threshold_min,
        N_threshold_max=N_threshold_max
    )
    def OnlineVanishingViscosityDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):
        
        @PreserveClassName
        class OnlineVanishingViscosityDecoratedProblem_Class(EllipticCoerciveProblem_DerivedClass):
            
            def __init__(self, V, **kwargs):
                # Store input parameters from the decorator factory
                self._viscosity = viscosity
                self._N_threshold_min = N_threshold_min
                self._N_threshold_max = N_threshold_max
                assert self._viscosity >= 0.
                assert self._N_threshold_min >= 0.
                assert self._N_threshold_max >= 0.
                assert self._N_threshold_min <= 1.
                assert self._N_threshold_max <= 1.
                assert self._N_threshold_min < self._N_threshold_max
                # Flag to enable or disable stabilization
                self.stabilized = True
                # Call to parent
                EllipticCoerciveProblem_DerivedClass.__init__(self, V, **kwargs)
                        
        # return value (a class) for the decorator
        return OnlineVanishingViscosityDecoratedProblem_Class
    
    # return the decorator itself
    return OnlineVanishingViscosityDecoratedProblem_Decorator
