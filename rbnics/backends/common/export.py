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
from rbnics.utils.decorators import backend_for, list_of, overload
from rbnics.utils.io import Folders, TextIO

NotImplementedType = type(NotImplemented)

@backend_for("common", inputs=((list_of(NotImplementedType), list_of(Number)), (Folders.Folder, str), str, (int, None), None))
def export(solution, directory, filename, suffix=None, component=None):
    _export(solution, directory, filename, suffix, component)
    
@overload(list_of(NotImplementedType), (Folders.Folder, str), str, (int, None), None)
def _export(solution, directory, filename, suffix=None, component=None): # used while trying to write out scalar outputs for a problem without any
    pass
    
@overload(list_of(Number), (Folders.Folder, str), str, (int, None), None)
def _export(solution, directory, filename, suffix=None, component=None):
    if suffix is not None:
        filename = filename + "_" + str(suffix)
    TextIO.save_file(solution, directory, filename)
