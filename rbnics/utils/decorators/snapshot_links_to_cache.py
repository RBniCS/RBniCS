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

import glob
import os
from rbnics.utils.mpi import parallel_io
from rbnics.utils.test import PatchInstanceMethod

def snapshot_links_to_cache(offline_method):
    def patched_export_solution(truth_problem, snapshots_folder):
        cache_folder = truth_problem.folder["cache"]
        original_export_solution = truth_problem.export_solution
        def patched_export_solution_internal(self_, folder=None, filename=None, *args, **kwargs):
            if str(folder) == str(snapshots_folder):
                assert (
                    hasattr(truth_problem, "_cache_file_from_kwargs")
                        or
                    hasattr(truth_problem, "_cache_file")
                )
                if hasattr(truth_problem, "_cache_file_from_kwargs"): # differential problem
                    cache_filename = truth_problem._cache_file_from_kwargs(**truth_problem._latest_solve_kwargs)
                elif hasattr(truth_problem, "_cache_file"): # EIM
                    cache_filename = truth_problem._cache_file()
                else:
                    raise AttributeError("Invalid cache file attribute.")
                def create_links():
                    for cache_path in glob.iglob(os.path.join(str(cache_folder), cache_filename + "*")):
                        cache_path_filename = os.path.basename(cache_path)
                        cache_relpath = os.path.join(os.path.relpath(str(cache_folder), str(folder)), cache_path_filename)
                        snapshot_path = os.path.join(str(folder), cache_path_filename.replace(cache_filename, filename))
                        if not os.path.exists(snapshot_path):
                            # Paraview output formats may require a light xml file that stores the path of a (possibly heavy)
                            # binary file. If the file is an xml file, we need to copy it and change the stored path.
                            # Otherwise, create a symbolic link.
                            try:
                                with open(cache_path, "r") as cache_file:
                                    header = cache_file.read(5)
                            except Exception:
                                should_link = True
                            else:
                                should_link = (header != "<?xml")
                            if should_link:
                                os.symlink(cache_relpath, snapshot_path)
                            else:
                                with open(cache_path, "r") as cache_file, open(snapshot_path, "w") as snapshot_file:
                                    for l in cache_file.readlines():
                                        snapshot_file.write(l.replace(cache_filename, filename))
                parallel_io(create_links)
            else:
                original_export_solution(folder, filename, *args, **kwargs)
        return patched_export_solution_internal
    
    def patched_offline_method(self_):
        # Patch truth_problem's export_solution
        assert (
            hasattr(self_, "truth_problem")
                or
            hasattr(self_, "EIM_approximation")
        )
        if hasattr(self_, "truth_problem"): # differential problem
            truth_problem = self_.truth_problem
        elif hasattr(self_, "EIM_approximation"): # EIM
            truth_problem = self_.EIM_approximation
        else:
            raise AttributeError("Invalid truth problem attribute.")
        export_solution_patch = PatchInstanceMethod(truth_problem, "export_solution", patched_export_solution(truth_problem, self_.folder["snapshots"]))
        export_solution_patch.patch()
        
        # Call standard offline
        reduced_problem = offline_method(self_)
        
        # Disable patch
        export_solution_patch.unpatch()
        
        # Return generated reduced problem
        return reduced_problem
        
    return patched_offline_method
