# Copyright (C) 2015-2018 by the RBniCS authors
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

import types
from rbnics.reduction_methods.base.reduction_method import ReductionMethod
from rbnics.utils.io import Folders
from rbnics.utils.decorators import UpdateMapFromProblemToTrainingStatus
from rbnics.utils.factories import ReducedProblemFactory

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
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
        all_folders = Folders()
        all_folders.update(self.folder)
        assert self.reduced_problem is not None
        all_folders.update(self.reduced_problem.folder)
        all_folders.pop("testing_set") # this is required only in the error/speedup analysis
        all_folders.pop("error_analysis") # this is required only in the error analysis
        all_folders.pop("speedup_analysis") # this is required only in the speedup analysis
        at_least_one_folder_created = all_folders.create()
        if not at_least_one_folder_created:
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
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
        # Create folder to store the results
        self.folder["speedup_analysis"].create()
        
        # Make sure to clean up problem and reduced problem solution cache to ensure that
        # solution and reduced solution are actually computed
        self.truth_problem._solution_cache.clear()
        self.reduced_problem._solution_cache.clear()
        self.truth_problem._output_cache.clear()
        self.reduced_problem._output_cache.clear()
        # ... and also disable the capability of importing/exporting truth solutions
        self._speedup_analysis__original_import_solution = self.truth_problem.import_solution
        def disabled_import_solution(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
            return False
        self.truth_problem.import_solution = types.MethodType(disabled_import_solution, self.truth_problem)
        self._speedup_analysis__original_export_solution = self.truth_problem.export_solution
        def disabled_export_solution(self_, folder=None, filename=None, solution=None, component=None, suffix=None):
            pass
        self.truth_problem.export_solution = types.MethodType(disabled_export_solution, self.truth_problem)
        
    # Finalize data structures required after the speedup analysis phase
    def _finalize_speedup_analysis(self, **kwargs):
        # Restore the capability to import/export truth solutions
        self.truth_problem.import_solution = self._speedup_analysis__original_import_solution
        del self._speedup_analysis__original_import_solution
        self.truth_problem.export_solution = self._speedup_analysis__original_export_solution
        del self._speedup_analysis__original_export_solution
