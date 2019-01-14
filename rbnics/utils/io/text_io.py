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

import os
from rbnics.utils.mpi import parallel_io

class TextIO(object):
    # Save a variable to file
    @staticmethod
    def save_file(content, directory, filename):
        if os.path.splitext(filename)[1] == "":
            filename = filename + ".txt"
        def save_file_task():
            with open(os.path.join(str(directory), filename), "w") as outfile:
                outfile.write(repr(content))
        parallel_io(save_file_task)
                
    # Load a variable from file
    @staticmethod
    def load_file(directory, filename, globals=None):
        if os.path.splitext(filename)[1] == "":
            filename = filename + ".txt"
        if globals is None:
            globals = dict()
        globals.update({"__builtins__": None})
        with open(os.path.join(str(directory), filename), "r") as infile:
            return eval(infile.read(), globals, {})
            
    # Check if the file exists
    @staticmethod
    def exists_file(directory, filename):
        if os.path.splitext(filename)[1] == "":
            filename = filename + ".txt"
        def exists_file_task():
            return os.path.exists(os.path.join(str(directory), filename))
        return parallel_io(exists_file_task)
