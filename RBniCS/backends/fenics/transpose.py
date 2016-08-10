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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.basic import transpose as basic_transpose
import RBniCS.backend.fenics
from RBniCS.backends.fenics.function import Function_Type
from RBniCS.backends.fenics.function_list import FunctionsList
from RBniCS.backends.fenics.vector import Vector_Type
import RBniCS.backend.fenics.wrapping
from RBniCS.utils.decorators import backend_for

@backend_for("FEniCS", online_backend="NumPy", inputs=any(Function_Type, FunctionsList, Vector_Type))
def transpose(arg):
    return basic_transpose(arg, RBniCS.backend.fenics, RBniCS.backend.fenics.wrapping)
    
