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


from rbnics.problems.base import LinearTimeDependentReducedProblem
from rbnics.backends import product, sum

def AbstractCFDUnsteadyReducedProblem(AbstractCFDUnsteadyReducedProblem_Base):
    return AbstractCFDUnsteadyReducedProblem_Base

def StokesUnsteadyReducedProblem(StokesReducedProblem_DerivedClass):

    StokesUnsteadyReducedProblem_Base = AbstractCFDUnsteadyReducedProblem(LinearTimeDependentReducedProblem(StokesReducedProblem_DerivedClass))

    class StokesUnsteadyReducedProblem_Class(StokesUnsteadyReducedProblem_Base):
            
        class ProblemSolver(StokesUnsteadyReducedProblem_Base.ProblemSolver):
            def residual_eval(self, t, solution, solution_dot):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("m", "a", "b", "bt", "f", "g"):
                    assert problem.terms_order[term] in (1, 2)
                    if problem.terms_order[term] == 2:
                        assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N, :N]))
                    elif problem.terms_order[term] == 1:
                        assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N]))
                    else:
                        raise ValueError("Invalid value for order of term " + term)
                return (
                      assembled_operator["m"]*solution_dot
                    + (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])*solution
                    - assembled_operator["f"] - assembled_operator["g"]
                )
                
            def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                for term in ("m", "a", "b", "bt"):
                    assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term][:N, :N]))
                return (
                      assembled_operator["m"]*solution_dot_coefficient
                    + assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
                )
        
    # return value (a class) for the decorator
    return StokesUnsteadyReducedProblem_Class
