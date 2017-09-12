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

from rbnics.problems.base import NonlinearReducedProblem
from rbnics.problems.base import ParametrizedReducedDifferentialProblem
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from rbnics.backends import product, sum
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import Extends

def NonlinearEllipticReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):
    
    NonlinearEllipticReducedProblem_Base = NonlinearReducedProblem(EllipticCoerciveReducedProblem_DerivedClass)
    
    @Extends(NonlinearEllipticReducedProblem_Base)
    class NonlinearEllipticReducedProblem_Class(NonlinearEllipticReducedProblem_Base):
        
        class ProblemSolver(NonlinearEllipticReducedProblem_Base.ProblemSolver):
            def residual_eval(self, solution):
                problem = self.problem
                N = self.N
                assembled_operator = dict()
                assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"][:N]))
                assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"][:N]))
                return assembled_operator["a"] - assembled_operator["f"]
                
            def jacobian_eval(self, solution):
                problem = self.problem
                N = self.N
                return sum(product(problem.compute_theta("da"), problem.operator["da"][:N, :N]))
        
        # Internal method for error computation. Unlike the linear case, do not use the energy norm.
        def _compute_error(self, **kwargs):
            # Call parent of parent (!), in order not to use the energy norm
            return ParametrizedReducedDifferentialProblem._compute_error(self, **kwargs)
            
        # Internal method for relative error computation. Unlike the linear case, do not use the energy norm.
        def _compute_relative_error(self, absolute_error, **kwargs):
            # Call parent of parent (!), in order not to use the energy norm
            return ParametrizedReducedDifferentialProblem._compute_relative_error(self, absolute_error, **kwargs)
        
    # return value (a class) for the decorator
    return NonlinearEllipticReducedProblem_Class

