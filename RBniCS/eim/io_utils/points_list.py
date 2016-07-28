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

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

from RBniCS.io_utils import ExportableList
from dolfin import Point

class PointsList(ExportableList):
    def __init__(self, mesh):
        ExportableList.__init__(self, "pickle")
        # Auxiliary list to store processor_id
        self.processors_id = list()
        # To get local points
        self.bounding_box_tree = mesh.bounding_box_tree()
        
    def load(self, directory, filename):
        return_value = ExportableList.load(self, directory, filename)
        # Make sure to update the processor ids
        for i in range(len(self)):
            self.processors_id.append(self._get_processor_id(ExportableList.__getitem__(self, i)))
        
    def append(self, point):
        ExportableList.append(self, point)
        # Make sure to update the processor ids
        self.processors_id.append(self._get_processor_id(point))
        
    def __getitem__(self, key):
        point = ExportableList.__getitem__(self, key)
        processor_id = self.processors_id[key]
        return (point, processor_id)
        
    def _get_processor_id(self, point):
        from mpi4py.MPI import MAX
        from RBniCS.io_utils import mpi_comm
        is_local = self.bounding_box_tree.collides_entity(Point(point))
        processor_id = -1
        if is_local:
            processor_id = mpi_comm.rank
        global_processor_id = mpi_comm.allreduce(processor_id, op=MAX)
        return global_processor_id
    
#  @}
########################### end - OFFLINE STAGE - end ########################### 

