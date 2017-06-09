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

from rbnics.utils.decorators import backend_for, list_of
from rbnics.utils.io import Folders, PickleIO

@backend_for("common", inputs=((list_of(float), list_of(int)), (Folders.Folder, str), str, None))
def import_(solution, directory, filename, suffix=None):
    if PickleIO.exists_file(directory, filename):
        loaded_solution = PickleIO.load_file(directory, filename)
        assert len(solution) == len(loaded_solution)
        for (i, solution_i) in enumerate(loaded_solution):
            solution[i] = float(solution_i)
        return True
    else:
        return False
        
    
