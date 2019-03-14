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
    'add_gold_options',
    'add_performance_options',
    'AttachInstanceMethod',
    'diff',
    'disable_matplotlib',
    'enable_logging',
    'enable_matplotlib',
    'load_tempdir',
    'patch_benchmark_plugin',
    'patch_initialize_testing_training_set',
    'PatchInstanceMethod',
    'process_gold_options',
    'run_and_compare_to_gold',
    'save_tempdir',
    'tempdir'
]
