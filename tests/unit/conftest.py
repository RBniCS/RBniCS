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
from rbnics.utils.test import disable_matplotlib, enable_matplotlib, load_tempdir, save_tempdir, tempdir  # noqa: F401


# Customize item selection
def pytest_collection_modifyitems(session, config, items):
    # Deselect first using markers (note: cannot import _pytest.mark globally)
    from _pytest.mark import pytest_collection_modifyitems as pytest_collection_modifyitems_from_marks
    pytest_collection_modifyitems_from_marks(items, config)

    # Separated parametrized forms tests require clean UFL and DOLFIN counters ...
    deselect_separated_parametrized_forms = False
    if any([item.name.startswith("test_separated_parametrized_forms") for item in items]):
        # ... so they cannot be mixed with other tests
        if not all([item.name.startswith("test_separated_parametrized_forms") for item in items]):
            deselect_separated_parametrized_forms = True
        # ... and with each other (scalar vs vector vs mixed)
        elif (
            any([item.name.startswith("test_separated_parametrized_forms_scalar") for item in items])
            + any([item.name.startswith("test_separated_parametrized_forms_vector") for item in items])
            + any([item.name.startswith("test_separated_parametrized_forms_mixed") for item in items])
            > 1
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
        ("test_linear_solver", "test_nonlinear_solver", "test_time_stepping"),
        ("test_mesh_to_submesh", "test_submesh_to_mesh", "test_submesh_global_cell_numbering_independent_on_mpi",
         "test_shared_entities_detection")
    ):
        if not all([any([item.name.startswith(test_matplotlib_prefix)
                         for test_matplotlib_prefix in test_matplotlib_prefixes]) for item in items]):
            for item in items:
                if any([item.name.startswith(test_matplotlib_prefix)
                        for test_matplotlib_prefix in test_matplotlib_prefixes]):
                    assert not hasattr(item, "_runtest_setup_function")
                    item._runtest_setup_function = disable_matplotlib
                    assert not hasattr(item, "_runtest_teardown_function")
                    item._runtest_teardown_function = enable_matplotlib


def pytest_runtest_setup(item):
    # Do the normal setup
    item.setup()
    # Carry out additional setup
    if hasattr(item, "_runtest_setup_function"):
        item._runtest_setup_function()


def pytest_runtest_teardown(item, nextitem):
    # Carry out additional teardown
    if hasattr(item, "_runtest_teardown_function"):
        item._runtest_teardown_function()
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    MPI.COMM_WORLD.Barrier()
