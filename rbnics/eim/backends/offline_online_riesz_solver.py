# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.backends import LinearSolver
from rbnics.backends.basic.wrapping import DelayedLinearSolver
from rbnics.eim.backends.offline_online_switch import OfflineOnlineSwitch

def OfflineOnlineRieszSolver(problem_name):
    if problem_name not in _offline_online_riesz_solver_cache:
        _OfflineOnlineRieszSolver_Base = OfflineOnlineSwitch(problem_name)
        class _OfflineOnlineRieszSolver(_OfflineOnlineRieszSolver_Base):
            def __call__(self, problem):
                return _OfflineOnlineRieszSolver._RieszSolver(problem, self._content[_OfflineOnlineRieszSolver_Base._current_stage])
                
            def attach(self, delay):
                if _OfflineOnlineRieszSolver_Base._current_stage not in self._content:
                    self._content[_OfflineOnlineRieszSolver_Base._current_stage] = delay
                else:
                    assert delay is self._content[_OfflineOnlineRieszSolver_Base._current_stage]
                    
            class _RieszSolver(object):
                def __init__(self, problem, delay):
                    self.problem = problem
                    self.delay = delay
                
                def solve(self, rhs):
                    problem = self.problem
                    args = (problem._riesz_solve_inner_product, problem._riesz_solve_storage, rhs, problem._riesz_solve_homogeneous_dirichlet_bc)
                    if not self.delay:
                        solver = LinearSolver(*args)
                        return solver.solve()
                    else:
                        return DelayedLinearSolver(*args)
                
        _offline_online_riesz_solver_cache[problem_name] = _OfflineOnlineRieszSolver
    
    return _offline_online_riesz_solver_cache[problem_name]
        
_offline_online_riesz_solver_cache = dict()
