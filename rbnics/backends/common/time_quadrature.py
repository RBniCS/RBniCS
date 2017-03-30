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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from scipy.integrate import simps
from rbnics.backends.abstract import TimeQuadrature as AbstractTimeQuadrature
from rbnics.utils.decorators import BackendFor, Extends, override, tuple_of

@Extends(AbstractTimeQuadrature)
@BackendFor("common", inputs=(tuple_of(float), float))
class TimeQuadrature(AbstractTimeQuadrature):
    def __init__(self, time_interval, time_step_size):
        self._time_step_size = time_step_size
        
    def integrate(self, function):
        return simps(function, dx=self._time_step_size)
        
