# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
try:
    import dolfin  # otherwise the next import from rbnics would disable dolfin as a required backend  # noqa: F401
except ImportError:
    pass
from rbnics.utils.test import (add_gold_options, disable_matplotlib, enable_matplotlib, PatchInstanceMethod,
                               process_gold_options, run_and_compare_to_gold)


def pytest_addoption(parser):
    add_gold_options(parser)


def pytest_configure(config):
    process_gold_options(config)


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        if item.name.startswith("test_eim_approximation_"):
            patch_test_item(item)


def patch_test_item(test_item):
    """
    Handle the execution of the test.
    """

    subdirectory = os.path.join(
        test_item.originalname + "_tempdir", test_item.callspec.getparam("expression_type"),
        test_item.callspec.getparam("basis_generation"))
    original_runtest = test_item.runtest

    @run_and_compare_to_gold(subdirectory)
    def runtest(self):
        disable_matplotlib()
        original_runtest()
        enable_matplotlib()

    PatchInstanceMethod(test_item, "runtest", runtest).patch()
