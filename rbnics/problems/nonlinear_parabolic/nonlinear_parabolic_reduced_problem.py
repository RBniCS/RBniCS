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

from rbnics.problems.base import NonlinearTimeDependentReducedProblem
from rbnics.backends import product, sum

def NonlinearParabolicReducedProblem(NonlinearEllipticReducedProblem_DerivedClass):
    
    NonlinearParabolicReducedProblem_Base = NonlinearTimeDependentReducedProblem(NonlinearEllipticReducedProblem_DerivedClass)
    
    class NonlinearParabolicReducedProblem_Class(NonlinearParabolicReducedProblem_Base):
            
        class ProblemSolver(NonlinearParabolicReducedProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["c"] = sum(product(problem.compute_theta("c"), problem.operator["c"][:N]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
                return (
                      assembled_operator["m"]*solution_dot
                    + assembled_operator["a"]*solution
                    + assembled_operator["c"]
                    - assembled_operator["f"]
                )
                
            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"][:N, :N]))
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                assembled_operator["dc"] = sum(product(problem.compute_theta("dc"), problem.operator["dc"][:N, :N]))
                return (
                      assembled_operator["m"]*solution_dot_coefficient
                    + assembled_operator["a"]
                    + assembled_operator["dc"]
                )
            
    # return value (a class) for the decorator
    return NonlinearParabolicReducedProblem_Class
