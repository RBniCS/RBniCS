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

from rbnics.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from rbnics.utils.decorators import Extends, override

@apply_decorator_only_once
def NonlinearTimeDependentPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):
    
    NonlinearTimeDependentPODGalerkinReduction_Base = TimeDependentPODGalerkinReduction(NonlinearPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass))
    
    @Extends(NonlinearTimeDependentPODGalerkinReduction_Base, preserve_class_name=True)
    class NonlinearTimeDependentPODGalerkinReduction_Class(NonlinearTimeDependentPODGalerkinReduction_Base):
        pass
                
    # return value (a class) for the decorator
    return NonlinearTimeDependentPODGalerkinReduction_Class
    
