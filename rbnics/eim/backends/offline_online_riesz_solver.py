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

from numbers import Number
from rbnics.backends import LinearSolver
from rbnics.backends.basic.wrapping import DelayedLinearSolver, DelayedProduct
from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch
from rbnics.utils.cache import cache
from rbnics.utils.decorators import overload

@cache
def OfflineOnlineRieszSolver(problem_name):
    _OfflineOnlineRieszSolver_Base = OfflineOnlineSwitch(problem_name)
    class _OfflineOnlineRieszSolver(_OfflineOnlineRieszSolver_Base):
        
        def __call__(self, problem):
            return _OfflineOnlineRieszSolver._RieszSolver(problem, self._content[_OfflineOnlineRieszSolver_Base._current_stage])
            
        def set_is_affine(self, is_affine):
            assert isinstance(is_affine, bool)
            if is_affine:
                delay = False
            else:
                delay = True
            if _OfflineOnlineRieszSolver_Base._current_stage not in self._content:
                self._content[_OfflineOnlineRieszSolver_Base._current_stage] = delay
            else:
                assert delay is self._content[_OfflineOnlineRieszSolver_Base._current_stage]
            
        def unset_is_affine(self):
            pass
            
        class _RieszSolver(object):
            def __init__(self, problem, delay):
                self.problem = problem
                self.delay = delay
            
            @overload
            def solve(self, rhs: object):
                problem = self.problem
                args = (problem._riesz_solve_inner_product, problem._riesz_solve_storage, rhs, problem._riesz_solve_homogeneous_dirichlet_bc)
                if not self.delay:
                    solver = LinearSolver(*args)
                    solver.set_parameters(problem._linear_solver_parameters)
                    solver.solve()
                    return problem._riesz_solve_storage
                else:
                    solver = DelayedLinearSolver(*args)
                    solver.set_parameters(problem._linear_solver_parameters)
                    return solver
                    
            @overload
            def solve(self, coef: Number, matrix: object, basis_function: object):
                if not self.delay:
                    rhs = coef*matrix*basis_function
                else:
                    rhs = DelayedProduct(coef)
                    rhs *= matrix
                    rhs *= basis_function
                return self.solve(rhs)
            
    return _OfflineOnlineRieszSolver
