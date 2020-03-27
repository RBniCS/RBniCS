# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import MPI
from rbnics.utils.test import add_performance_options, patch_benchmark_plugin

def pytest_addoption(parser):
    add_performance_options(parser)

def pytest_configure(config):
    assert config.pluginmanager.hasplugin("benchmark")
    patch_benchmark_plugin(config.pluginmanager.getplugin("benchmark"))

def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    MPI.barrier(MPI.comm_world)
