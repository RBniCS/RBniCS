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

from rbnics.backends import product, sum
from rbnics.utils.decorators import Extends
from rbnics.problems.base import LinearReducedProblem

def GeostrophicReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    GeostrophicReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)
    
    @Extends(GeostrophicReducedProblem_Base)
    class GeostrophicReducedProblem_Class(GeostrophicReducedProblem_Base):
    
        class ProblemSolver(GeostrophicReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                
            def vector_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
        
    # return value (a class) for the decorator
    return GeostrophicReducedProblem_Class
