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

import types
from rbnics.reduction_methods.base.reduction_method import ReductionMethod
from rbnics.utils.io import Folders
from rbnics.utils.decorators import Extends, override, UpdateMapFromProblemToTrainingStatus
from rbnics.utils.factories import ReducedProblemFactory

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ReductionMethod) # needs to be first in order to override for last the methods.
@UpdateMapFromProblemToTrainingStatus
class DifferentialProblemReductionMethod(ReductionMethod):
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ReductionMethod.__init__(self, type(truth_problem).__name__, truth_problem.mu_range)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Reduced order problem
        self.reduced_problem = None
        self._init_kwargs = kwargs
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.truth_problem = truth_problem
    
    ## Initialize data structures required for the offline phase
    @override
    def _init_offline(self):
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem = ReducedProblemFactory(self.truth_problem, self, **self._init_kwargs)
        
        # Prepare folders and init reduced problem
        all_folders = Folders()
        all_folders.update(self.folder)
        all_folders.update(self.reduced_problem.folder)
        all_folders.pop("testing_set") # this is required only in the error analysis
        at_least_one_folder_created = all_folders.create()
        if not at_least_one_folder_created:
            self.reduced_problem.init("online")
            return False # offline construction should be skipped, since data are already available
        else:
            self.reduced_problem.init("offline")
            return True # offline construction should be carried out
            
    ## Finalize data structures required after the offline phase
    @override
    def _finalize_offline(self):
        self.reduced_problem.init("online")
    
    ## Initialize data structures required for the error analysis phase
    @override
    def _init_error_analysis(self, **kwargs): 
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
    ## Initialize data structures required for the speedup analysis phase
    @override
    def _init_speedup_analysis(self, **kwargs): 
        # Initialize the affine expansion in the truth problem
        self.truth_problem.init()
        
        # Initialize reduced order data structures in the reduced problem
        self.reduced_problem.init("online")
        
        # Make sure to clean up problem and reduced problem solution cache to ensure that
        # solution and reduced solution are actually computed
        self.truth_problem._solution_cache.clear()
        self.reduced_problem._solution_cache.clear()
        # ... and also disable the capability of importing truth solutions
        self._speedup_analysis__original_import_solution = self.truth_problem.import_solution
        def disabled_import_solution(self_, folder, filename, solution=None):
            return False
        self.truth_problem.import_solution = types.MethodType(disabled_import_solution, self.truth_problem)
        
    ## Finalize data structures required after the speedup analysis phase
    @override
    def _finalize_speedup_analysis(self, **kwargs):
        # Restore the capability to import truth solutions
        self.truth_problem.import_solution = self._speedup_analysis__original_import_solution
        del self._speedup_analysis__original_import_solution
    
