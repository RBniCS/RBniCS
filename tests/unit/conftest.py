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

import pytest
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
from dolfin_utils.test import tempdir
matplotlib_backend = matplotlib.get_backend()

# Customize item selection
def pytest_collection_modifyitems(session, config, items):
    # Deselect first using markers
    from _pytest.mark import pytest_collection_modifyitems as pytest_collection_modifyitems_from_marks # cannot import globally
    pytest_collection_modifyitems_from_marks(items, config)
    
    # Separated parametrized forms tests require clean UFL and DOLFIN counters ...
    deselect_separated_parametrized_forms = False
    if any([item.name.startswith("test_separated_parametrized_forms") for item in items]):
        # ... so they cannot be mixed with other tests
        if not all([item.name.startswith("test_separated_parametrized_forms") for item in items]):
            deselect_separated_parametrized_forms = True
        # ... and with each other (scalar vs vector vs mixed)
        elif (
            (
                any([item.name.startswith("test_separated_parametrized_forms_scalar") for item in items])
                    +
                any([item.name.startswith("test_separated_parametrized_forms_vector") for item in items])
                    +
                any([item.name.startswith("test_separated_parametrized_forms_mixed") for item in items])
            ) > 1
        ):
            deselect_separated_parametrized_forms = True
    if deselect_separated_parametrized_forms:
        selected_items = list()
        deselected_items = list()
        for item in items:
            if item.name.startswith("test_separated_parametrized_forms"):
                deselected_items.append(item)
            else:
                selected_items.append(item)
        config.hook.pytest_deselected(items=deselected_items)
        items[:] = selected_items
        
    # IO tests are split in "save", "load" and "io" (that saves and load at the same time) in order to
    # possibly use different number of mpi processes at saving and loading time. If the "io" one is enabled,
    # than disable the "save" and "load" ones.
    for test_io_prefix in (
        "test_tensor",
        "test_reduced_mesh"
    ):
        if any([item.name.startswith(test_io_prefix + "_io") for item in items]):
            selected_items = list()
            deselected_items = list()
            for item in items:
                if item.name.startswith(test_io_prefix + "_save") or item.name.startswith(test_io_prefix + "_load"):
                    deselected_items.append(item)
                else:
                    selected_items.append(item)
            config.hook.pytest_deselected(items=deselected_items)
            items[:] = selected_items
            
    # Some tests open by default plots using matplotlib. Disable interactive plots
    # unless only those specific tests were enabled
    for test_matplotlib_prefixes in (
        ("test_sampling", ),
        ("test_time_stepping", ),
        ("test_mesh_to_submesh", "test_submesh_to_mesh", "test_submesh_global_cell_numbering_independent_on_mpi", "test_shared_entities_detection")
    ):
        if not all([any([item.name.startswith(test_matplotlib_prefix) for test_matplotlib_prefix in test_matplotlib_prefixes]) for item in items]):
            for item in items:
                if any([item.name.startswith(test_matplotlib_prefix) for test_matplotlib_prefix in test_matplotlib_prefixes]):
                    assert not hasattr(item, "_runtest_setup_function")
                    item._runtest_setup_function = disable_matplotlib
                    assert not hasattr(item, "_runtest_teardown_function")
                    item._runtest_teardown_function = enable_matplotlib
            
def pytest_runtest_setup(item):
    if hasattr(item, "_runtest_setup_function"):
        item._runtest_setup_function()
        
def pytest_runtest_teardown(item, nextitem):
    if hasattr(item, "_runtest_teardown_function"):
        item._runtest_teardown_function()
            
# Helper functions to enable and disable matplotlib
def disable_matplotlib():
    plt.switch_backend("agg")
    
def enable_matplotlib():
    plt.switch_backend(matplotlib_backend)
    plt.close("all") # do not trigger matplotlib max_open_warning
    
# Temporarily change the tempdir fixture to avoid it clearing out the temporary folder
os_mkdir = os.mkdir
shutil_rmtree = shutil.rmtree

def do_not_rmtree(arg):
    pass

def mkdir_for_save(arg):
    os_mkdir(arg.replace("_test_tensor_save", "test_tensor_io"))
    
def mkdir_for_load(arg):
    pass
    
@pytest.fixture(scope="function")
def save_tempdir(request):
    function_name = request.function.__name__
    request.function.__name__ = function_name.replace("_save", "_io")
    os.mkdir = mkdir_for_save
    output = tempdir(request)
    request.function.__name__ = function_name
    os.mkdir = os_mkdir
    return output

@pytest.fixture(scope="function")
def load_tempdir(request):
    function_name = request.function.__name__
    request.function.__name__ = function_name.replace("_load", "_io")
    os.mkdir = mkdir_for_load
    shutil.rmtree = do_not_rmtree
    output = tempdir(request)
    request.function.__name__ = function_name
    os.mkdir = os_mkdir
    shutil.rmtree = shutil_rmtree
    return output
