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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     OFFLINE AND ONLINE COMMON INTERFACES     ########################### 
## @defgroup OfflineOnlineInterfaces Common interfaces for offline and online
#  @{

from RBniCS.linear_algebra.truth_vector import TruthVector
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector_Base as OnlineVector
from RBniCS.linear_algebra.online_matrix import OnlineMatrix_Base as OnlineMatrix

# Similarly to FEniCS' solve define a solve for online problems
def solve(A, x, b, bcs):
    if isinstance(A, TruthMatrix) and isinstance(x, TruthVector) and isinstance(b, TruthVector):
        for bc in bcs:
            bc.apply(A, b)
        from dolfin import solve as dolfin_solve
        dolfin_solve(A, x, b)
    elif isinstance(A, OnlineMatrix) and isinstance(x, OnlineVector) and isinstance(b, OnlineVector):
        for i in range(bcs):
            b[i] = bcs[i]
            A[i, :] = 0.
            A[i, i] = 1.
        from numpy.linalg import solve as numpy_solve
        x = numpy_solve(A, b)
    else:
        raise RuntimeError("Invalid input arguments in solve")
        
#  @}
########################### end - OFFLINE AND ONLINE COMMON INTERFACES - end ########################### 
