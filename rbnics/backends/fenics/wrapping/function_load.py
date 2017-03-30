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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path
from dolfin import File
from rbnics.utils.mpi import is_io_process

def function_load(fun, directory, filename, suffix=None):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    full_filename = str(directory) + "/" + filename + ".xml"
    file_exists = False
    if is_io_process() and os.path.exists(full_filename):
        file_exists = True
    file_exists = is_io_process.mpi_comm.bcast(file_exists, root=is_io_process.root)
    if file_exists:
        file = File(full_filename)
        file >> fun
    return file_exists
    
