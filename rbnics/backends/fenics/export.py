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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.backends.basic import export as basic_export
import rbnics.backends.fenics
from rbnics.backends.fenics.function import Function
from rbnics.backends.fenics.matrix import Matrix
from rbnics.backends.fenics.vector import Vector
import rbnics.backends.fenics.wrapping
from rbnics.utils.decorators import backend_for
from rbnics.utils.io import Folders

# Export a solution to file
@backend_for("fenics", inputs=((Function.Type(), Matrix.Type(), Vector.Type()), (Folders.Folder, str), str, (int, None), (int, str, None)))
def export(solution, directory, filename, suffix=None, component=None):
    basic_export(solution, directory, filename, suffix, component, rbnics.backends.fenics, rbnics.backends.fenics.wrapping)
    
