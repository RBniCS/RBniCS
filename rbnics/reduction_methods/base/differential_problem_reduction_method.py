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

import inspect
from rbnics.backends import assign
from rbnics.reduction_methods.base.reduction_method import ReductionMethod
from rbnics.utils.io import Folders
from rbnics.utils.decorators import StoreMapFromProblemToReductionMethod, UpdateMapFromProblemToTrainingStatus
from rbnics.utils.factories import ReducedProblemFactory
from rbnics.utils.test import PatchInstanceMethod

@StoreMapFromProblemToReductionMethod
@UpdateMapFromProblemToTrainingStatus
class DifferentialProblemReductionMethod(ReductionMethod):
    
    # Default initialization of members
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ReductionMethod.__init__(self, truth_problem.name())
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Reduced order problem
        self.reduced_problem = None
        self._init_kwargs = kwargs
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.truth_problem = truth_problem
        
    def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
        return ReductionMethod.initialize_training_set(self, self.truth_problem.mu_range, ntrain, enable_import, sampling, **kwargs)
        
    def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
        return ReductionMethod.initialize_testing_set(self, self.truth_problem.mu_range, ntest, enable_import, sampling, **kwargs)
    
    # Initialize data structures required for the offline phase
    def _init_offline(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem = ReducedProblemFactory(self.truth_problem, self, **self._init_kwargs)
        
        # Prepare folders and init reduced problem
        required_folders = Folders()
        required_folders.update(self.folder)
        assert self.reduced_problem is not None
        required_folders.update(self.truth_problem.folder)
        required_folders.update(self.reduced_problem.folder)
        optional_folders = Folders()
        optional_folders["cache"] = required_folders.pop("cache") # this does not affect the availability of offline data
        optional_folders["testing_set"] = required_folders.pop("testing_set") # this is required only in the error/speedup analysis
        optional_folders["error_analysis"] = required_folders.pop("error_analysis") # this is required only in the error analysis
        optional_folders["speedup_analysis"] = required_folders.pop("speedup_analysis") # this is required only in the speedup analysis
        at_least_one_required_folder_created = required_folders.create()
        at_least_one_optional_folder_created = optional_folders.create()  # noqa: F841
        if not at_least_one_required_folder_created:
            return False # offline construction should be skipped, since data are already available
        else:
            self.reduced_problem.init("offline")
            return True # offline construction should be carried out
        
    def postprocess_snapshot(self, snapshot, snapshot_index):
        """
        Postprocess a snapshot before adding it to the basis/snapshot matrix, for instance removing non-homogeneous Dirichlet boundary conditions.
        
        :param snapshot: truth offline solution.
        :param snapshot_index: truth offline solution index.
        """
        n_components = len(self.reduced_problem.components)
        # Get helper strings and functions depending on the number of basis components
        if n_components > 1:
            dirichlet_bc_string = "dirichlet_bc_{c}"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.reduced_problem.dirichlet_bc[component] and not self.reduced_problem.dirichlet_bc_are_homogeneous[component]
            def assert_lengths(component, theta_bc):
                assert self.reduced_problem.N_bc[component] == len(theta_bc)
        else:
            dirichlet_bc_string = "dirichlet_bc"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.reduced_problem.dirichlet_bc and not self.reduced_problem.dirichlet_bc_are_homogeneous
            def assert_lengths(component, theta_bc):
                assert self.reduced_problem.N_bc == len(theta_bc)
        # Carry out postprocessing
        for component in self.reduced_problem.components:
            if has_non_homogeneous_dirichlet_bc(component):
                theta_bc = self.reduced_problem.compute_theta(dirichlet_bc_string.format(c=component))
                assert_lengths(component, theta_bc)
                return snapshot - self.reduced_problem.basis_functions[:self.reduced_problem.N_bc]*theta_bc
            else:
                return snapshot
            
    # Finalize data structures required after the offline phase
    def _finalize_offline(self):
        self.reduced_problem.init("online")
    
    # Initialize data structures required for the error analysis phase
    def _init_error_analysis(self, **kwargs):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
        # Create folder to store the results
        self.folder["error_analysis"].create()
        
        # Patch truth solve if with_respect_to kwarg is provided
        self._patch_truth_solve(False, **kwargs)
        
        # Patch truth compute_output if with_respect_to kwarg is provided
        self._patch_truth_compute_output(False, **kwargs)
        
    def _finalize_error_analysis(self, **kwargs):
        # Undo patch to truth solve in case with_respect_to kwarg was provided
        self._undo_patch_truth_solve(False, **kwargs)
        
        # Undo patch to truth compute_output in case with_respect_to kwarg was provided
        self._undo_patch_truth_compute_output(False, **kwargs)
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()

        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
        # Create folder to store the results
        self.folder["speedup_analysis"].create()
        
        # Patch truth solve if with_respect_to kwarg is provided
        self._patch_truth_solve(True, **kwargs)
        
        # Patch truth compute_output if with_respect_to kwarg is provided
        self._patch_truth_compute_output(True, **kwargs)
        
    # Finalize data structures required after the speedup analysis phase
    def _finalize_speedup_analysis(self, **kwargs):
        # Undo patch to truth solve in case with_respect_to kwarg was provided
        self._undo_patch_truth_solve(True, **kwargs)
        
        # Undo patch to truth compute_output in case with_respect_to kwarg was provided
        self._undo_patch_truth_compute_output(True, **kwargs)
        
    def _patch_truth_solve(self, force, **kwargs):
        if "with_respect_to" in kwargs:
            assert inspect.isfunction(kwargs["with_respect_to"])
            other_truth_problem = kwargs["with_respect_to"](self.truth_problem)
            def patched_truth_solve(self_, **kwargs_):
                other_truth_problem.solve(**kwargs_)
                assign(self.truth_problem._solution, other_truth_problem._solution)
                return self.truth_problem._solution
                
            self.patch_truth_solve = PatchInstanceMethod(
                self.truth_problem,
                "solve",
                patched_truth_solve
            )
            self.patch_truth_solve.patch()
            
            # Initialize the affine expansion in the other truth problem
            other_truth_problem.init()
        else:
            other_truth_problem = self.truth_problem
            
        # Clean up solution caching and disable I/O
        if force:
            # Make sure to clean up problem and reduced problem solution cache to ensure that
            # solution and reduced solution are actually computed
            other_truth_problem._solution_cache.clear()
            self.reduced_problem._solution_cache.clear()
            
            # Disable the capability of importing/exporting truth solutions
            def disable_import_solution_method(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
                raise OSError
            self.disable_import_solution = PatchInstanceMethod(other_truth_problem, "import_solution", disable_import_solution_method)
            self.disable_import_solution.patch()
            def disable_export_solution_method(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
                pass
            self.disable_export_solution = PatchInstanceMethod(other_truth_problem, "export_solution", disable_export_solution_method)
            self.disable_export_solution.patch()
        
    def _undo_patch_truth_solve(self, force, **kwargs):
        if "with_respect_to" in kwargs:
            self.patch_truth_solve.unpatch()
            del self.patch_truth_solve
            
        # Restore solution I/O
        if force:
            # Restore the capability to import/export truth solutions
            self.disable_import_solution.unpatch()
            self.disable_export_solution.unpatch()
            del self.disable_import_solution
            del self.disable_export_solution
            
    def _patch_truth_compute_output(self, force, **kwargs):
        if "with_respect_to" in kwargs:
            assert inspect.isfunction(kwargs["with_respect_to"])
            other_truth_problem = kwargs["with_respect_to"](self.truth_problem)
            def patched_truth_compute_output(self_):
                other_truth_problem.compute_output()
                self.truth_problem._output = other_truth_problem._output
                return self.truth_problem._output
                
            self.patch_truth_compute_output = PatchInstanceMethod(
                self.truth_problem,
                "compute_output",
                patched_truth_compute_output
            )
            self.patch_truth_compute_output.patch()
            
            # Initialize the affine expansion in the other truth problem
            other_truth_problem.init()
        else:
            other_truth_problem = self.truth_problem
            
        # Clean up output caching and disable I/O
        if force:
            # Make sure to clean up problem and reduced problem output cache to ensure that
            # output and reduced output are actually computed
            other_truth_problem._output_cache.clear()
            self.reduced_problem._output_cache.clear()
            
            # Disable the capability of importing/exporting truth outputs
            def disable_import_output_method(self_, folder=None, filename=None, output=None, suffix=None):
                raise OSError
            self.disable_import_output = PatchInstanceMethod(other_truth_problem, "import_output", disable_import_output_method)
            self.disable_import_output.patch()
            def disable_export_output_method(self_, folder=None, filename=None, output=None, suffix=None):
                pass
            self.disable_export_output = PatchInstanceMethod(other_truth_problem, "export_output", disable_export_output_method)
            self.disable_export_output.patch()
        
    def _undo_patch_truth_compute_output(self, force, **kwargs):
        if "with_respect_to" in kwargs:
            self.patch_truth_compute_output.unpatch()
            del self.patch_truth_compute_output
            
        # Restore output I/O
        if force:
            # Restore the capability to import/export truth outputs
            self.disable_import_output.unpatch()
            self.disable_export_output.unpatch()
            del self.disable_import_output
            del self.disable_export_output
