# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.test.patch_initialize_testing_training_set import patch_initialize_testing_training_set


def add_gold_options(parser):
    available_options = [name for opt in parser._anonymous.options for name in opt.names()]
    # Comparison to gold files in methodology tests and tutorials
    if "--action" not in available_options:
        parser.addoption("--action", action="store", default=None)
    if "--data-dir" not in available_options:
        parser.addoption("--data-dir", action="store", default=None)


def add_performance_options(parser):
    available_options = [name for opt in parser._anonymous.options for name in opt.names()]
    # Comparison to previous performance tests
    if "--overhead-speedup-storage" not in available_options:
        parser.addoption("--overhead-speedup-storage", action="store", default=".benchmarks")


def process_gold_options(config):
    if config.option.action is not None:
        patch_initialize_testing_training_set(config.option.action)
