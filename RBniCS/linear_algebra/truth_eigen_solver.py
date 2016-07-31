# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file truth_eigen_solver.py
#  @brief Type of the truth eigen solver
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

from RBniCS.utils.decorators import Extends, override

# Declare truth eigen solver type (from FEniCS)
from dolfin import SLEPcEigenSolver, as_backend_type
@Extends(SLEPcEigenSolver)
class TruthEigenSolver(SLEPcEigenSolver):
    @override
    def __init__(self, A=None, B=None):
        if A is not None:
            A = as_backend_type(A)
        if B is not None:
            B = as_backend_type(B)
        SLEPcEigenSolver.__init__(self, A, B)
    
#  @}
########################### end - OFFLINE STAGE - end ########################### 

