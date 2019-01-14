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

import hashlib
from rbnics.problems.base import LinearTimeDependentProblem
from rbnics.problems.stokes import StokesProblem
from rbnics.backends import copy, product, sum

def AbstractCFDUnsteadyProblem(AbstractCFDUnsteadyProblem_Base):
    class AbstractCFDUnsteadyProblem_Class(AbstractCFDUnsteadyProblem_Base):
        
        # Default initialization of members
        def __init__(self, V, **kwargs):
            # Call to parent
            AbstractCFDUnsteadyProblem_Base.__init__(self, V, **kwargs)
            
            # Form names for saddle point problems
            self.terms.append("m")
            self.terms_order.update({"m": 2})
            
        def solve_supremizer(self, solution):
            return copy(AbstractCFDUnsteadyProblem_Base.solve_supremizer(self, solution))
            
        def _solve_supremizer(self, solution):
            print("# t = {0:g}".format(self.t))
            AbstractCFDUnsteadyProblem_Base._solve_supremizer(self, solution)
            
        def _supremizer_integer_index(self):
            try:
                monitor_t0 = self._time_stepping_parameters["monitor"]["initial_time"]
            except KeyError:
                monitor_t0 = self.t0
            try:
                monitor_dt = self._time_stepping_parameters["monitor"]["time_step_size"]
            except KeyError:
                assert self.dt is not None
                monitor_dt = self.dt
            return int(round((self.t - monitor_t0)/monitor_dt))
            
        def _supremizer_cache_key_from_kwargs(self, **kwargs):
            cache_key = AbstractCFDUnsteadyProblem_Base._supremizer_cache_key_from_kwargs(self, **kwargs)
            cache_key += (self._supremizer_integer_index(), )
            return cache_key
            
        def _supremizer_cache_file_from_kwargs(self, **kwargs):
            return hashlib.sha1(str(AbstractCFDUnsteadyProblem_Base._supremizer_cache_key_from_kwargs(self, **kwargs)).encode("utf-8")).hexdigest()
            
        def export_supremizer(self, folder=None, filename=None, supremizer=None, component=None, suffix=None):
            assert suffix is None
            AbstractCFDUnsteadyProblem_Base.export_supremizer(self, folder, filename, supremizer=supremizer, component=component, suffix=self._supremizer_integer_index())
            
        def import_supremizer(self, folder=None, filename=None, supremizer=None, component=None, suffix=None):
            assert suffix is None
            AbstractCFDUnsteadyProblem_Base.import_supremizer(self, folder, filename, supremizer=supremizer, component=component, suffix=self._supremizer_integer_index())

        def export_solution(self, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
            if component is None:
                component = ["u", "p"] # but not "s"
            AbstractCFDUnsteadyProblem_Base.export_solution(self, folder, filename, solution_over_time, component, suffix)
            
        def import_solution(self, folder=None, filename=None, solution_over_time=None, component=None, suffix=None):
            if component is None:
                component = ["u", "p"] # but not "s"
            AbstractCFDUnsteadyProblem_Base.import_solution(self, folder, filename, solution_over_time, component, suffix)
            
    return AbstractCFDUnsteadyProblem_Class
        
# Base class containing the definition of saddle point problems
StokesUnsteadyProblem_Base = AbstractCFDUnsteadyProblem(LinearTimeDependentProblem(StokesProblem))

class StokesUnsteadyProblem(StokesUnsteadyProblem_Base):
    class ProblemSolver(StokesUnsteadyProblem_Base.ProblemSolver):
        def residual_eval(self, t, solution, solution_dot):
            problem = self.problem
            assembled_operator = dict()
            for term in ("m", "a", "b", "bt", "f", "g"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["m"]*solution_dot
                + (assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"])*solution
                - assembled_operator["f"] - assembled_operator["g"]
            )
            
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            problem = self.problem
            assembled_operator = dict()
            for term in ("m", "a", "b", "bt"):
                assembled_operator[term] = sum(product(problem.compute_theta(term), problem.operator[term]))
            return (
                  assembled_operator["m"]*solution_dot_coefficient
                + assembled_operator["a"] + assembled_operator["b"] + assembled_operator["bt"]
            )
