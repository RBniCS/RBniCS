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

from rbnics.problems.base import NonlinearTimeDependentProblem
from rbnics.problems.navier_stokes import NavierStokesProblem
from rbnics.problems.stokes_unsteady.stokes_unsteady_problem import AbstractCFDUnsteadyProblem
from rbnics.backends import product, sum

NavierStokesUnsteadyProblem_Base = AbstractCFDUnsteadyProblem(NonlinearTimeDependentProblem(NavierStokesProblem))

# Base class containing the definition of saddle point problems
class NavierStokesUnsteadyProblem(NavierStokesUnsteadyProblem_Base):
    class ProblemSolver(NavierStokesUnsteadyProblem_Base.ProblemSolver):
        def residual_eval(self, t, solution, solution_dot):
            problem = self.problem
            assembled_operator = dict()
            for term in ("m", "a", "b", "bt", "c", "f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["m"]*solution_dot
                + (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])*solution
                + assembled_operator["c"]
                - assembled_operator["f"] - assembled_operator["g"]
            )
            
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            problem = self.problem
            assembled_operator = dict()
            for term in ("m", "a", "b", "bt", "dc"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["m"]*solution_dot_coefficient
                + assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                + assembled_operator["dc"]
            )
