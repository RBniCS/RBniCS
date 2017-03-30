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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import hashlib
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.utils.decorators import Extends, override, sync_setters

@Extends(EIMApproximation)
class TimeDependentEIMApproximation(EIMApproximation):
    
    @override
    @sync_setters("truth_problem", "set_time", "t")
    @sync_setters("truth_problem", "set_time_step_size", "dt")
    @sync_setters("truth_problem", "set_final_time", "T")
    def __init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation):
        # Call the parent initialization
        EIMApproximation.__init__(self, truth_problem, parametrized_expression, folder_prefix, basis_generation)
        
        # Store quantities related to the time discretization
        self.t = 0.
        self.dt = None
        self.T  = None
        
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
    
    ## Set the current value of the parameter. The parameter may have been extended in internal EIM classes to include also time.
    def set_mu(self, mu):
        assert isinstance(mu, (tuple, EnlargedMu))
        if isinstance(mu, tuple):
            EIMApproximation.set_mu(self, mu)
        else:
            assert len(mu) == 2
            assert "mu" in mu
            assert isinstance(mu["mu"], tuple)
            EIMApproximation.set_mu(self, mu["mu"])
            assert "t" in mu
            assert isinstance(mu["t"], float)
            self.set_time(mu["t"])
            
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
        
