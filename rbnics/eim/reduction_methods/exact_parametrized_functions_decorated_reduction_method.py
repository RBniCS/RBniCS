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

from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.eim.problems import ExactParametrizedFunctions

@ReductionMethodDecoratorFor(ExactParametrizedFunctions)
def ExactParametrizedFunctionsDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    @PreserveClassName
    class ExactParametrizedFunctionsDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
        
        def set_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.set_training_set(self, ntrain, enable_import, sampling, **kwargs)
            # Since exact evaluation is required, we cannot use a distributed training set
            self.training_set.serialize_maximum_computations()
            return import_successful
            
    # return value (a class) for the decorator
    return ExactParametrizedFunctionsDecoratedReductionMethod_Class
