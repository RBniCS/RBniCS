# Copyright (C) 2015-2019 by the RBniCS authors
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

from numbers import Number
from rbnics.utils.decorators import backend_for, list_of
from rbnics.utils.io import Folders, TextIO

@backend_for("common", inputs=(list_of(Number), (Folders.Folder, str), str, (int, None), None))
def import_(solution, directory, filename, suffix=None, component=None):
    if suffix is not None:
        filename = filename + "_" + str(suffix)
    if TextIO.exists_file(directory, filename):
        loaded_solution = TextIO.load_file(directory, filename)
        assert len(solution) == len(loaded_solution)
        for (i, solution_i) in enumerate(loaded_solution):
            solution[i] = float(solution_i)
    else:
        raise OSError
