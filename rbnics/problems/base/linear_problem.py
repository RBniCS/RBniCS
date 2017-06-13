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

from rbnics.backends import assign, LinearProblemWrapper, LinearSolver
from rbnics.utils.decorators import Extends, override, RequiredBaseDecorators

@RequiredBaseDecorators(None)
def LinearProblem(ParametrizedDifferentialProblem_DerivedClass):
    
    @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class LinearProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        
        class ProblemSolver(ParametrizedDifferentialProblem_DerivedClass.ProblemSolver, LinearProblemWrapper):
            def solve(self):
                problem = self.problem
                solver = LinearSolver(self.matrix_eval(), problem._solution, self.vector_eval(), self.bc_eval())
                solver.solve()
                
    # return value (a class) for the decorator
    return LinearProblem_Class
    
