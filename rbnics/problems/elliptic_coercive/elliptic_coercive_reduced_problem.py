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

from rbnics.problems.base import LinearReducedProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.backends import product, sum, transpose
from rbnics.backends.online import OnlineFunction
from rbnics.reduction_methods.elliptic_coercive import EllipticCoerciveReductionMethod

def EllipticCoerciveReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    
    EllipticCoerciveReducedProblem_Base = LinearReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass)

    # Base class containing the interface of a projection based ROM
    # for elliptic coercive problems.
    class EllipticCoerciveReducedProblem_Class(EllipticCoerciveReducedProblem_Base):
        
        class ProblemSolver(EllipticCoerciveReducedProblem_Base.ProblemSolver):
            def matrix_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("a"), problem.operator["a"][:N, :N]))
                
            def vector_eval(self):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
            
        # Perform an online evaluation of the (compliant) output
        def _compute_output(self, N):
            self._output = transpose(self._solution)*sum(product(self.compute_theta("f"), self.operator["f"][:N]))
        
        # Internal method for error computation
        def _compute_error(self, **kwargs):
            inner_product = dict()
            inner_product["u"] = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm
            assert "inner_product" not in kwargs
            kwargs["inner_product"] = inner_product
            return EllipticCoerciveReducedProblem_Base._compute_error(self, **kwargs)
            
        # Internal method for relative error computation
        def _compute_relative_error(self, absolute_error, **kwargs):
            inner_product = dict()
            inner_product["u"] = sum(product(self.truth_problem.compute_theta("a"), self.truth_problem.operator["a"])) # use the energy norm
            assert "inner_product" not in kwargs
            kwargs["inner_product"] = inner_product
            return EllipticCoerciveReducedProblem_Base._compute_relative_error(self, absolute_error, **kwargs)
        
    # return value (a class) for the decorator
    return EllipticCoerciveReducedProblem_Class
        
