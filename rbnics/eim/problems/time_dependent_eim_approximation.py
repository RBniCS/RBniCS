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

import hashlib
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.utils.decorators import Extends, sync_setters

def set_mu_decorator(set_mu):
    def decorated_set_mu(self, mu):
        assert isinstance(mu, (tuple, EnlargedMu))
        if isinstance(mu, tuple):
            set_mu(self, mu)
        else:
            assert len(mu) == 2
            assert "mu" in mu
            assert isinstance(mu["mu"], tuple)
            set_mu(self, mu["mu"])
            assert "t" in mu
            assert isinstance(mu["t"], float)
            self.set_time(mu["t"])
    return decorated_set_mu

@Extends(EIMApproximation)
class TimeDependentEIMApproximation(EIMApproximation):
    
    @sync_setters("truth_problem", "set_mu", "mu", set_mu_decorator)
    @sync_setters("truth_problem", "set_time", "t")
    @sync_setters("truth_problem", "set_initial_time", "t0")
    @sync_setters("truth_problem", "set_time_step_size", "dt")
    @sync_setters("truth_problem", "set_final_time", "T")
    def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
        # Call the parent initialization
        EIMApproximation.__init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation)
        
        # Store quantities related to the time discretization
        self.t0 = 0.
        self.t = 0.
        self.dt = None
        self.T  = None
        
    ## Set initial time
    def set_initial_time(self, t0):
        assert isinstance(t0, (float, int))
        t0 = float(t0)
        self.t0 = t0
        
    ## Set current time
    def set_time(self, t):
        assert isinstance(t, (float, int))
        t = float(t)
        self.t = t
        
    ## Set time step size
    def set_time_step_size(self, dt):
        assert isinstance(dt, (float, int))
        dt = float(dt)
        self.dt = dt
        
    ## Set final time
    def set_final_time(self, T):
        assert isinstance(T, (float, int))
        T = float(T)
        self.T = T
            
    def _cache_key_and_file(self):
        cache_key = (self.mu, self.t)
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        return (cache_key, cache_file)

class EnlargedMu(dict):
    def __str__(self):
        assert len(self) == 2
        assert "mu" in self
        assert isinstance(self["mu"], tuple)
        assert "t" in self
        assert isinstance(self["t"], float)
        output = str(self["mu"]) + " and t = " + str(self["t"])
        return output
        
