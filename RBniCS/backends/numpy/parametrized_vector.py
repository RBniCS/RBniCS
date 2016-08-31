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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.abstract import ParametrizedVector as AbstractParametrizedVector
from RBniCS.backends.numpy.vector import Vector
from RBniCS.utils.decorators import BackendFor, Extends, override

@Extends(AbstractParametrizedVector)
@BackendFor("NumPy", inputs=(Vector.Type(), ))
class ParametrizedVector(AbstractParametrizedVector):
    def __init__(self, vector):
        AbstractParametrizedVector.__init__(vector)
        #
        self._vector = vector
        self._mpi_comm = None #TODO
    
    @override
    @property
    def vector(self):
        return self._vector
        
    @override
    def get_processor_id(self, indices):
        return self._mpi_comm.rank # vector is repeated on all processors
        
