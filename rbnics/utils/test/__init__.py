# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.test.attach_instance_method import AttachInstanceMethod
from rbnics.utils.test.diff import diff
from rbnics.utils.test.enable_logging import enable_logging
from rbnics.utils.test.matplotlib import disable_matplotlib, enable_matplotlib
from rbnics.utils.test.options import add_gold_options, add_performance_options, process_gold_options
from rbnics.utils.test.patch_benchmark_plugin import patch_benchmark_plugin
from rbnics.utils.test.patch_initialize_testing_training_set import patch_initialize_testing_training_set
from rbnics.utils.test.patch_instance_method import PatchInstanceMethod
from rbnics.utils.test.run_and_compare_to_gold import run_and_compare_to_gold
from rbnics.utils.test.tempdir import load_tempdir, save_tempdir, tempdir

__all__ = [
    "add_gold_options",
    "add_performance_options",
    "AttachInstanceMethod",
    "diff",
    "disable_matplotlib",
    "enable_logging",
    "enable_matplotlib",
    "load_tempdir",
    "patch_benchmark_plugin",
    "patch_initialize_testing_training_set",
    "PatchInstanceMethod",
    "process_gold_options",
    "run_and_compare_to_gold",
    "save_tempdir",
    "tempdir"
]
