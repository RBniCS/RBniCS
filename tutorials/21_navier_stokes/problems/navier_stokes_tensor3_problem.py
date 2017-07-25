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
            "a", "b", "bt", "c", "f", "g",
            "da", "db", "dbt", "dc",
            # Auxiliary terms for supremizer enrichment
            "bt_restricted"
        ]
        self.terms_order = {
            "a": 1, "b": 1, "bt": 1, "c": 1, "f": 1, "g": 1,
            "da": 2, "db": 2, "dbt": 2, "dc": 2,
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
                  assembled_operator["a"] + assembled_operator["c"]
                + assembled_operator["b"] + assembled_operator["bt"]
                - assembled_operator["f"] - assembled_operator["g"]
            )
            
        def jacobian_eval(self, solution):
            self.store_solution(solution)
            problem = self.problem
            assembled_operator = dict()
            for term in ("da", "db", "dbt", "dc"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["da"] + assembled_operator["dc"]
                + assembled_operator["db"] + assembled_operator["dbt"]
            )
            
    def _solve_supremizer(self):
        assert len(self.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.inner_product["s"][0]
        assembled_operator_rhs = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"]))
        if self.dirichlet_bc["s"] is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc_s"), self.dirichlet_bc["s"]))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator_lhs,
            self._supremizer,
            assembled_operator_rhs,
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._supremizer
    
