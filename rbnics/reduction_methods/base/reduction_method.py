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

import os
from abc import ABCMeta, abstractmethod
from rbnics.sampling import ParameterSpaceSubset
from rbnics.utils.io import Folders

# Implementation of a class containing an offline/online decomposition of ROM for parametrized problems
class ReductionMethod(object, metaclass=ABCMeta):
    def __init__(self, folder_prefix):
        # I/O
        self.folder_prefix = folder_prefix
        self.folder = Folders()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # Maximum reduced order space dimension to be used for the stopping criterion in the basis selection
        self.Nmax = 0
        # Tolerance to be used for the stopping criterion in the basis selection
        self.tol = 0.
        # Training set
        self.training_set = ParameterSpaceSubset()
        # I/O
        self.folder["training_set"] = os.path.join(self.folder_prefix, "training_set")
        
        # $$ ERROR ANALYSIS AND SPEEDUP ANALYSIS DATA STRUCTURES $$ #
        # Testing set
        self.testing_set = ParameterSpaceSubset()
        # I/O
        self.folder["testing_set"] = os.path.join(self.folder_prefix, "testing_set")
        self.folder["error_analysis"] = os.path.join(self.folder_prefix, "error_analysis")
        self.folder["speedup_analysis"] = os.path.join(self.folder_prefix, "speedup_analysis")
    
    # OFFLINE: set maximum reduced space dimension (stopping criterion)
    def set_Nmax(self, Nmax, **kwargs):
        self.Nmax = Nmax
        
    # OFFLINE: set tolerance (stopping criterion)
    def set_tolerance(self, tol, **kwargs):
        self.tol = tol

    # OFFLINE: set the elements in the training set.
    def initialize_training_set(self, mu_range, ntrain, enable_import=True, sampling=None, **kwargs):
        # Create I/O folder
        self.folder["training_set"].create()
        # Test if can import
        import_successful = False
        if enable_import:
            try:
                self.training_set.load(self.folder["training_set"], "training_set")
            except OSError:
                import_successful = False
            else:
                import_successful = (len(self.training_set) == ntrain)
        if not import_successful:
            self.training_set.generate(mu_range, ntrain, sampling)
            # Export
            self.training_set.save(self.folder["training_set"], "training_set")
        return import_successful
        
    # ERROR ANALYSIS: set the elements in the testing set.
    def initialize_testing_set(self, mu_range, ntest, enable_import=False, sampling=None, **kwargs):
        # Create I/O folder
        self.folder["testing_set"].create()
        # Test if can import
        import_successful = False
        if enable_import:
            try:
                self.testing_set.load(self.folder["testing_set"], "testing_set")
            except OSError:
                import_successful = False
            else:
                import_successful = (len(self.testing_set) == ntest)
        if not import_successful:
            self.testing_set.generate(mu_range, ntest, sampling)
            # Export
            self.testing_set.save(self.folder["testing_set"], "testing_set")
        return import_successful
    
    # Perform the offline phase of the reduced order model
    @abstractmethod
    def offline(self):
        raise NotImplementedError("Please implement the offline phase of the reduced order model.")
        
    # Initialize data structures required for the offline phase
    def _init_offline(self):
        pass
        
    # Finalize data structures required after the offline phase
    def _finalize_offline(self):
        pass
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the testing set
    @abstractmethod
    def error_analysis(self, N_generator=None, filename=None, **kwargs):
        raise NotImplementedError("Please implement the error analysis of the reduced order model.")
        
    # Initialize data structures required for the error analysis phase
    def _init_error_analysis(self, **kwargs):
        pass
        
    # Finalize data structures required after the error analysis phase
    def _finalize_error_analysis(self, **kwargs):
        pass
        
    # Compute the speedup analysis of the reduced order approximation with respect to the full order one
    # over the testing set
    @abstractmethod
    def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
        raise NotImplementedError("Please implement the speedup analysis of the reduced order model.")
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        pass
        
    # Finalize data structures required after the speedup analysis phase
    def _finalize_speedup_analysis(self, **kwargs):
        pass
