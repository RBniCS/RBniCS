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

from rbnics.utils.io import NumpyIO

def function_load(fun, directory, filename, suffix=None):
    if suffix is not None:
        filename = filename + "." + str(suffix)
    file_exists = NumpyIO.exists_file(directory, filename)
    if file_exists:
        vec = NumpyIO.load_file(directory, filename)
        fun.vector()[:] = vec
    return file_exists
