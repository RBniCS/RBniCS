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

from rbnics.problems.base import LinearReducedProblem
from rbnics.backends import product, sum, transpose

def EllipticReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    EllipticReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    # Base class containing the interface of a projection based ROM
    # for elliptic problems.
    class EllipticReducedProblem_Class(EllipticReducedProblem_Base):
        
        class ProblemSolver(EllipticReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                
            def vector_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
            
        # Perform an online evaluation of the output
        def _compute_output(self, N):
            self._output = transpose(self._solution)*sum(product(self.compute_theta("s"), self.operator["s"][:N]))
            
    # return value (a class) for the decorator
    return EllipticReducedProblem_Class
