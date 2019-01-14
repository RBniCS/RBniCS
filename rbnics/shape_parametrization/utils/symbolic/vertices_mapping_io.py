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

from rbnics.utils.io import TextIO

class VerticesMappingIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(vertices_mapping, directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        TextIO.save_file(vertices_mapping, directory, filename)
    
    # Load a variable from file
    @staticmethod
    def load_file(directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        return TextIO.load_file(directory, filename)
            
    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if not filename.endswith(".vmp"):
            filename = filename + ".vmp"
        return TextIO.exists_file(directory, filename)
