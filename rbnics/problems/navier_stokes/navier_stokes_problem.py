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
from rbnics.problems.stokes import StokesProblem
from rbnics.backends import product, sum

NavierStokesProblem_Base = NonlinearProblem(StokesProblem)

class NavierStokesProblem(NavierStokesProblem_Base):
    
    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call to parent
        NavierStokesProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for Navier-Stokes problems
        self.terms.extend([
            "c", "dc"
        ])
        self.terms_order.update({
            "c": 1, "dc": 2
        })
        
    class ProblemSolver(NavierStokesProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "c", "f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])*solution
                 + assembled_operator["c"]
                 - assembled_operator["f"] - assembled_operator["g"]
            )
            
        def jacobian_eval(self, solution):
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "dc"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                + assembled_operator["dc"]
            )
