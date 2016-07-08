# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file reduction_method.py
#  @brief Implementation of a class containing an offline/online decomposition for ROM for parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os # for path and makedir
from RBniCS.sampling import ParameterSpaceSubset

#~~~~~~~~~~~~~~~~~~~~~~~~~     REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReductionMethod
#
# Implementation of a class containing an offline/online decomposition of ROM for parametrized problems
class ReductionMethod(object):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, folder_prefix, mu_range):
        # I/O
        self.folder_prefix = folder_prefix
        self.folder = dict() # from string to string
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Maximum reduced order space dimension to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        # Training set
        self.xi_train = ParameterSpaceSubset()
        # I/O
        self.folder["xi_train"] = self.folder_prefix + "/" + "xi_train"
        
        # $$ ERROR ANALYSIS DATA STRUCTURES $$ #
        # Test set
        self.xi_test = ParameterSpaceSubset()
        # I/O
        self.folder["xi_test" ] = self.folder_prefix + "/" + "xi_test"
        
        # $$ OFFLINE/ERROR ANALYSIS DATA STRUCTURES $$ #
        self._mu_range = mu_range # local copy to generate training/test sets,
                                  # should not be used anywhere else
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set maximum reduced space dimension (stopping criterion)
    def set_Nmax(self, Nmax, **kwargs):
        self.Nmax = Nmax

    ## OFFLINE: set the elements in the training set \xi_train.
    def set_xi_train(self, ntrain, enable_import=True, sampling=None):
        # Create I/O folder
        if not os.path.exists(self.folder["xi_train"]):
            os.makedirs(self.folder["xi_train"])
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_train.load(self.folder["xi_train"], "xi_train") \
                and  (len(self.xi_train) == ntrain)
        if not import_successful:
            self.xi_train.generate(self._mu_range, ntrain, sampling)
            # Export 
            self.xi_train.save(self.folder["xi_train"], "xi_train")
        
    ## ERROR ANALYSIS: set the elements in the test set \xi_test.
    def set_xi_test(self, ntest, enable_import=False, sampling=None):
        # Create I/O folder
        if not os.path.exists(self.folder["xi_test"]):
            os.makedirs(self.folder["xi_test"])
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_test.load(self.folder["xi_test"], "xi_test") \
                and  (len(self.xi_test) == ntest)
        if not import_successful:
            self.xi_test.generate(self._mu_range, ntest, sampling)
            # Export 
            self.xi_test.save(self.folder["xi_test"], "xi_test")
            
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        raise RuntimeError("Please implement the offline phase of the reduced order model.")
        
    ## Initialize data structures required for the offline phase
    def _init_offline(self):
        pass
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
        
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        raise RuntimeError("Please implement the error analysis of the reduced order model.")
        
    ## Initialize data structures required for the error analysis phase
    def _init_error_analysis(self):
        pass
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    

