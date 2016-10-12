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
## @file online_matrix.py
#  @brief Type of online matrix
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import matrix

class _Matrix_Type(matrix): # inherit to make sure that matrices and vectors correspond to two different types
    pass
    
from numpy import zeros as _MatrixContent_Base
from RBniCS.utils.decorators import backend_for, OnlineSizeType

@backend_for("NumPy", inputs=(OnlineSizeType, OnlineSizeType), output=_Matrix_Type)
def Matrix(M, N):
    assert isinstance(M, (int, dict))
    assert isinstance(N, (int, dict))
    assert isinstance(M, dict) == isinstance(N, dict)
    if isinstance(M, dict):
        M_sum = sum(M.values())
        N_sum = sum(N.values())
    else:
        M_sum = M
        N_sum = N
    output = _Matrix_Type(_MatrixContent_Base((M_sum, N_sum)))
    output.M = M
    output.N = N
    return output
    
