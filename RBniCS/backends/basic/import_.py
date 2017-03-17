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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path
from RBniCS.utils.mpi import is_io_process

# Returns True if it was possible to import the file, False otherwise
def import_(solution, directory, filename, suffix, backend, wrapping):
    return_value = False
    if is_io_process() and os.path.exists(directory + "/" + filename):
        return_value = True
    return_value = is_io_process.mpi_comm.bcast(return_value, root=is_io_process.root)
    
    if return_value:
        assert isinstance(solution, (backend.Function.Type(), backend.Matrix.Type(), backend.Vector.Type()))
        if isinstance(solution, backend.Function.Type()):
            wrapping.function_load(solution, directory, filename, suffix=suffix)
        elif isinstance(solution, (backend.Matrix.Type(), backend.Vector.Type())):
            assert suffix is None
            wrapping.tensor_load(solution, directory, filename)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in export.")
        
    return return_value
