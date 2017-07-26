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
from rbnics.problems.stokes import StokesProblem
from rbnics.backends import LinearSolver, product, sum
from rbnics.utils.decorators import Extends, override
from rbnics.utils.mpi import log, PROGRESS

NavierStokesProblem_Base = NonlinearProblem(StokesProblem)

@Extends(StokesProblem)
class NavierStokesProblem(NavierStokesProblem_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        NavierStokesProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for Navier-Stokes problems
        self.terms = [
            "f", "g", "c",
            "a", "b", "bt", "dc",
            # Auxiliary terms for supremizer enrichment
            "bt_restricted"
        ]
        self.terms_order = {
            "f": 1, "g": 1, "c": 1,
            "a": 2, "b": 2, "bt": 2, "dc": 2,
            # Auxiliary terms for supremizer enrichment
            "bt_restricted": 2
        }
        
    class ProblemSolver(NavierStokesProblem_Base.ProblemSolver):
        def residual_eval(self, solution):
            self.store_solution(solution)
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "c", "f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])*solution
                 - assembled_operator["f"] - assembled_operator["g"] + assembled_operator["c"]
            )
            
        def jacobian_eval(self, solution):
            self.store_solution(solution)
            problem = self.problem
            assembled_operator = dict()
            for term in ("a", "b", "bt", "dc"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["a"] + assembled_operator["dc"]
                + assembled_operator["b"] + assembled_operator["bt"]
            )
    
