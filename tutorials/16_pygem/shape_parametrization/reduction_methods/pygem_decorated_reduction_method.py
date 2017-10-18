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

from rbnics.sampling.distributions import UniformDistribution
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from sampling import DiscardInadmissibleDeformations
from shape_parametrization.problems import PyGeM

@ReductionMethodDecoratorFor(PyGeM)
def PyGeMDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    @PreserveClassName
    class PyGeMDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
        
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            if sampling is None:
                sampling = DiscardInadmissibleDeformations(UniformDistribution)(self.truth_problem)
            return DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            
        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            if sampling is None:
                sampling = DiscardInadmissibleDeformations(UniformDistribution)(self.truth_problem)
            return DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
            
    # return value (a class) for the decorator
    return PyGeMDecoratedReductionMethod_Class
