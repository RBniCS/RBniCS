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

from abc import ABCMeta, abstractmethod
from RBniCS.sampling import ParameterSpaceSubset
from RBniCS.utils.io import Folders

#~~~~~~~~~~~~~~~~~~~~~~~~~     REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReductionMethod
#
# Implementation of a class containing an offline/online decomposition of ROM for parametrized problems
class ReductionMethod(object):
    __metaclass__ = ABCMeta
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, folder_prefix, mu_range):
        # I/O
        self.folder_prefix = folder_prefix
        self.folder = Folders()
        
        assert len(mu_range) > 0
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Maximum reduced order space dimension to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        # Training set
        self.xi_train = ParameterSpaceSubset(mu_range)
        # I/O
        self.folder["xi_train"] = self.folder_prefix + "/" + "xi_train"
        
        # $$ ERROR ANALYSIS DATA STRUCTURES $$ #
        # Test set
        self.xi_test = ParameterSpaceSubset(mu_range)
        # I/O
        self.folder["xi_test" ] = self.folder_prefix + "/" + "xi_test"
    
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
        self.folder["xi_train"].create()
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_train.load(self.folder["xi_train"], "xi_train") \
                and (len(self.xi_train) == ntrain)
        if not import_successful:
            self.xi_train.generate(ntrain, sampling)
            # Export 
            self.xi_train.save(self.folder["xi_train"], "xi_train")
        return import_successful
        
    ## ERROR ANALYSIS: set the elements in the test set \xi_test.
    def set_xi_test(self, ntest, enable_import=False, sampling=None):
        # Create I/O folder
        self.folder["xi_test"].create()
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_test.load(self.folder["xi_test"], "xi_test") \
                and  (len(self.xi_test) == ntest)
        if not import_successful:
            self.xi_test.generate(ntest, sampling)
            # Export 
            self.xi_test.save(self.folder["xi_test"], "xi_test")
        return import_successful
            
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    @abstractmethod
    def offline(self):
        raise NotImplementedError("Please implement the offline phase of the reduced order model.")
        
    ## Initialize data structures required for the offline phase
    @abstractmethod
    def _init_offline(self):
        raise NotImplementedError("Please implement the initialization for offline phase of the reduced order model.")
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
        
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    @abstractmethod
    def error_analysis(self, N=None):
        raise NotImplementedError("Please implement the error analysis of the reduced order model.")
        
    ## Initialize data structures required for the error analysis phase
    @abstractmethod
    def _init_error_analysis(self):
        raise NotImplementedError("Please implement the initialization for error analysis of the reduced order model.")
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    

