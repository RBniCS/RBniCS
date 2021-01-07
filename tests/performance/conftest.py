# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from mpi4py import MPI
try:
    import dolfin  # otherwise the next import from rbnics would disable dolfin as a required backend  # noqa: F401
except ImportError:
    pass
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
    MPI.COMM_WORLD.Barrier()
