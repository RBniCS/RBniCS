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


from logging import CRITICAL, DEBUG, ERROR, getLogger, INFO, log, WARNING
from rbnics.utils.mpi.parallel_io import parallel_io
from rbnics.utils.mpi.parallel_max import parallel_max
from rbnics.utils.mpi.print import print

def set_log_level(log_level):
    getLogger().setLevel(log_level)

__all__ = [
    'CRITICAL',
    'DEBUG',
    'ERROR',
    'INFO',
    'log',
    'parallel_io',
    'parallel_max',
    'print',
    'set_log_level',
    'WARNING'
]
