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
## @file pickle_io.py
#  @brief I/O helper functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     I/O     ########################### 
## @defgroup IO Input/output methods
#  @{

import pickle
import os # for path
from RBniCS.io_utils.mpi import mpi_comm

class PickleIO(object):
    
    ## Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        with open(str(directory) + "/" + filename + ".pkl", "rb") as infile:
            return pickle.load(infile)
    
    ## Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if mpi_comm.rank == 0:
            with open(str(directory) + "/" + filename + ".pkl", "wb") as outfile:
                pickle.dump(content, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        mpi_comm.barrier()
            
    ## Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        exists = None
        if mpi_comm.rank == 0:
            exists = os.path.exists(str(directory) + "/" + filename + ".pkl")
        exists = mpi_comm.bcast(exists, root=0)
        return exists
        
#  @}
########################### end - I/O - end ########################### 

