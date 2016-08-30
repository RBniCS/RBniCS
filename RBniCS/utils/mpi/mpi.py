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
## @file mpi.py
#  @brief Basic mpi configuration
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     I/O     ########################### 
## @defgroup IO Input/output methods
#  @{

try:
    from mpi4py import MPI
except ImportError:
    raise # TODO
else:
    from dolfin import mpi_comm_world as dolfin_mpi_comm_world
    _default_io_mpi_comm = dolfin_mpi_comm_world().tompi4py()
    del dolfin_mpi_comm_world
    
    # I/O operations should be carried out only on processor 0
    def is_io_process(mpi_comm=None):
        if mpi_comm is None:
            is_io_process.mpi_comm = _default_io_mpi_comm
            return _default_io_mpi_comm.rank == 0
        else:
            is_io_process.mpi_comm = None # the user already has the mpi_comm available
            return mpi_comm.rank == 0
            
    is_io_process.root = 0
    is_io_process.mpi_comm = _default_io_mpi_comm
        
#  @}
########################### end - I/O - end ########################### 

