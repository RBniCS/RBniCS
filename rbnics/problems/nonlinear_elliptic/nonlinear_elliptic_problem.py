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

from rbnics.problems.base import NonlinearProblem
from rbnics.problems.elliptic_coercive import EllipticCoerciveProblem
from rbnics.backends import product, sum
from rbnics.utils.decorators import Extends, override

NonlinearEllipticProblem_Base = NonlinearProblem(EllipticCoerciveProblem)

@Extends(NonlinearEllipticProblem_Base)
class NonlinearEllipticProblem(NonlinearEllipticProblem_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        NonlinearEllipticProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for nonlinear problems
        self.terms = ["a", "da", "f"]
        self.terms_order = {"a": 1, "da": 2, "f": 1}
    
    class ProblemSolver(NonlinearEllipticProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            return assembled_operator["a"] - assembled_operator["f"]
            
        def jacobian_eval(self, solution):
            problem = self.problem
            return sum(product(problem.compute_theta("da"), problem.operator["da"]))
    
