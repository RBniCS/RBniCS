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

from rbnics.problems.base import LinearProblem, ParametrizedDifferentialProblem
from rbnics.backends import product, sum, transpose

EllipticProblem_Base = LinearProblem(ParametrizedDifferentialProblem)

# Base class containing the definition of elliptic problems
class EllipticProblem(EllipticProblem_Base):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        EllipticProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for elliptic problems
        self.terms = ["a", "f", "s"]
        self.terms_order = {"a": 2, "f": 1, "s": 1}
        self.components = ["u"]
    
    class ProblemSolver(EllipticProblem_Base.ProblemSolver):
        def matrix_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("a"), problem.operator["a"]))
            
        def vector_eval(self):
            problem = self.problem
            return sum(product(problem.compute_theta("f"), problem.operator["f"]))
            
    # Perform a truth evaluation of the output
    def _compute_output(self):
        self._output = transpose(self._solution)*sum(product(self.compute_theta("s"), self.operator["s"]))
