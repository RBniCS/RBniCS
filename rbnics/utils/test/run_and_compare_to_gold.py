# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import shutil
import glob
try:
    import git
except ImportError:
    # Full git support is only required by regold action;
    # null and compare actions do not require any git support
    pass
from mpi4py.MPI import COMM_WORLD
from rbnics.utils.mpi import parallel_io
from rbnics.utils.test.diff import diff


def run_and_compare_to_gold(subdirectory=""):

    def run_and_compare_to_gold_decorator(runtest):

        def run_and_compare_to_gold_function(self):
            """
            Handles the comparison of test/tutorial with gold files
            """

            mpi_comm = COMM_WORLD

            rootdir = str(self.config.rootdir)

            # Get action
            action = self.config.option.action
            assert action in ("compare", "regold", None)

            # Get data directory
            if action is not None:
                data_dir = self.config.option.data_dir
                assert data_dir is not None
            else:
                data_dir = None

            # Get current and reference directory
            current_dir = str(self.fspath.dirname)
            if action is not None:
                reference_dir = os.path.join(
                    current_dir.replace(rootdir, data_dir), self.fspath.basename, str(mpi_comm.size))
                current_dir = os.path.join(current_dir, subdirectory)
                reference_dir = os.path.join(reference_dir, subdirectory)
            else:
                reference_dir = None

            # Copy training and testing sets
            if action is not None:
                def copy_training_and_testing_sets():
                    for set_ in ("testing_set", "training_set"):
                        set_directories = glob.glob(os.path.join(reference_dir, "**", set_), recursive=True)
                        if action == "compare":
                            assert len(set_directories) > 0
                        for set_directory in set_directories:
                            set_directory = os.path.relpath(set_directory, reference_dir)
                            if os.path.exists(os.path.join(reference_dir, set_directory)):
                                if os.path.exists(os.path.join(current_dir, set_directory)):
                                    shutil.rmtree(os.path.join(current_dir, set_directory))
                                shutil.copytree(os.path.join(reference_dir, set_directory),
                                                os.path.join(current_dir, set_directory))
                parallel_io(copy_training_and_testing_sets, mpi_comm)

            # Run test/tutorial
            runtest(self)

            # Process results
            def process_results():
                if action == "compare":
                    failures = list()
                    filenames = glob.glob(os.path.join(reference_dir, "**", "*.*"), recursive=True)
                    assert len(filenames) > 0
                    for filename in filenames:
                        filename = os.path.relpath(filename, reference_dir)
                        diffs = diff(os.path.join(reference_dir, filename), os.path.join(current_dir, filename))
                        if len(diffs) > 0:
                            failures.append(filename)
                            os.makedirs(os.path.dirname(os.path.join(current_dir, filename + "_diff")), exist_ok=True)
                            with open(os.path.join(current_dir, filename + "_diff"), "w") as failure_file:
                                failure_file.writelines(diffs)
                    if len(failures) > 0:
                        raise RuntimeError(
                            self.name + ", comparison has failed for the following files: " + str(failures) + ".")
                elif action == "regold":
                    data_dir_repo = git.Repo(data_dir)
                    assert not data_dir_repo.is_dirty()
                    # Move current files to reference directory
                    if os.path.exists(reference_dir):
                        shutil.rmtree(reference_dir)
                    shutil.copytree(current_dir, reference_dir)
                    if os.path.exists(os.path.join(reference_dir, ".gitignore")):
                        os.remove(os.path.join(reference_dir, ".gitignore"))
                    data_dir_repo.git.add([reference_dir])
                    # Commit changes
                    commit = str(git.Repo(rootdir).head.commit)
                    relpath = os.path.relpath(str(self.fspath), rootdir)
                    if self.name != relpath:
                        message = ("Automatic regold of " + self.name + " in " + relpath
                                   + " at upstream commit " + commit)
                    else:
                        message = "Automatic regold of " + relpath + " at upstream commit " + commit
                    data_dir_repo.git.commit(message=message)
                    # Clean repository
                    data_dir_repo.git.clean("-Xdf")
            parallel_io(process_results, mpi_comm)

        return run_and_compare_to_gold_function

    return run_and_compare_to_gold_decorator
