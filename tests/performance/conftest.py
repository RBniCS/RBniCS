# Copyright (C) 2015-2018 by the RBniCS authors
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

from dolfin import MPI
from rbnics.utils.test import add_performance_options, patch_benchmark_plugin
from dolfin import has_pybind11 # added back to dolfin as a side effect of rbnics import
if not has_pybind11():
    from dolfin import mpi_comm_world

def pytest_addoption(parser):
    add_performance_options(parser)

def pytest_configure(config):
    assert config.pluginmanager.hasplugin("benchmark")
    patch_benchmark_plugin(config.pluginmanager.getplugin("benchmark"))
    
def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    if has_pybind11():
        MPI.barrier(MPI.comm_world)
    else:
        MPI.barrier(mpi_comm_world())
