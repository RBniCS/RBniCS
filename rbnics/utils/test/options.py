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

import os
import tempfile
try:
    import git
except ImportError:
    # Minimal support to be able to do a git clone
    class git(object):
        class Repo(object):
            @classmethod
            def clone_from(cls, url, to_path):
                os.system("git clone " + url + " " + to_path)
from rbnics.utils.test.patch_initialize_testing_training_set import patch_initialize_testing_training_set

def add_gold_options(parser):
    available_options = [name for opt in parser._anonymous.options for name in opt.names()]
    # Comparison to gold files in methodology tests and tutorials
    if "--action" not in available_options:
        parser.addoption("--action", action="store", default=None)
    if "--data-dir" not in available_options:
        parser.addoption("--data-dir", action="store", default=os.environ.get("RBNICS_TEST_DATA", None))
        
def add_performance_options(parser):
    available_options = [name for opt in parser._anonymous.options for name in opt.names()]
    # Comparison to previous performance tests
    if "--overhead-speedup-storage" not in available_options:
        parser.addoption("--overhead-speedup-storage", action="store", default=".benchmarks")
        
def process_gold_options(config):
    if config.option.action is not None:
        if config.option.data_dir is None:
            assert config.option.action is not "regold", "Please provide a data directory"
            data_dir = tempfile.mkdtemp()
            git.Repo.clone_from("git@gitlab.com:RBniCS/RBniCS-test-data.git", data_dir)
            config.option.data_dir = data_dir
        patch_initialize_testing_training_set(config.option.action)
