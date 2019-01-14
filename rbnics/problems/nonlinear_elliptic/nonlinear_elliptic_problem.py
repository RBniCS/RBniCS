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

from rbnics.problems.base import NonlinearProblem
from rbnics.problems.elliptic import EllipticProblem
from rbnics.backends import product, sum

NonlinearEllipticProblem_Base = NonlinearProblem(EllipticProblem)

class NonlinearEllipticProblem(NonlinearEllipticProblem_Base):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NonlinearEllipticProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for nonlinear problems
        self.terms = ["a", "c", "dc", "f"]
        self.terms_order = {"a": 2, "c": 1, "dc": 2, "f": 1}
    
    class ProblemSolver(NonlinearEllipticProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            return assembled_operator["a"]*solution + assembled_operator["c"] - assembled_operator["f"]
            
        def jacobian_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"]))
            return assembled_operator["a"] + assembled_operator["dc"]
