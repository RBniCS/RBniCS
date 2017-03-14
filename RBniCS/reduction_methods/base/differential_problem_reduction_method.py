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
## @file elliptic_coercive_reduction_method.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.reduction_methods.base.reduction_method import ReductionMethod
from RBniCS.utils.io import Folders
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.factories import ReducedProblemFactory

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReductionMethodBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ReductionMethod) # needs to be first in order to override for last the methods.
class DifferentialProblemReductionMethod(ReductionMethod):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
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
            
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
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
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
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
        
        # Make sure to clean up reduced problem solution cache to ensure that
        # reduced solution are actually computed
        self.reduced_problem._solution_cache.clear()
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
