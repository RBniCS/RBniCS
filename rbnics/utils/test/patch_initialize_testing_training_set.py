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

def patch_initialize_testing_training_set(action):
    import rbnics.reduction_methods.base
    import rbnics.eim.reduction_methods
    
    """
    Patch ReductionMethod.initialize_{testing,training}_set to always read from file if
    action == "compare", and to try first to read from file if actions == "regold"
    """
    if action is None:
        pass
    elif action == "compare":
        def initialize_training_set(self, mu_range, ntrain, enable_import=True, sampling=None, **kwargs):
            try:
                self.training_set.load(self.folder["training_set"], "training_set")
            except OSError:
                raise AssertionError("Loading should have not been failing")
            return True
        rbnics.reduction_methods.base.ReductionMethod.initialize_training_set = initialize_training_set
            
        def initialize_testing_set(self, mu_range, ntest, enable_import=False, sampling=None, **kwargs):
            try:
                self.testing_set.load(self.folder["testing_set"], "testing_set")
            except OSError:
                raise AssertionError("Loading should have not been failing")
            return True
        rbnics.reduction_methods.base.ReductionMethod.initialize_testing_set = initialize_testing_set
    elif action == "regold":
        original_initialize_training_set = rbnics.reduction_methods.base.ReductionMethod.initialize_training_set
        def initialize_training_set(self, mu_range, ntrain, enable_import=True, sampling=None, **kwargs):
            self.folder["training_set"].create()
            try:
                self.training_set.load(self.folder["training_set"], "training_set")
            except OSError:
                return original_initialize_training_set(self, mu_range, ntrain, enable_import, sampling, **kwargs)
            else:
                return True
        rbnics.reduction_methods.base.ReductionMethod.initialize_training_set = initialize_training_set
        
        original_initialize_testing_set = rbnics.reduction_methods.base.ReductionMethod.initialize_testing_set
        def initialize_testing_set(self, mu_range, ntest, enable_import=False, sampling=None, **kwargs):
            self.folder["testing_set"].create()
            try:
                self.testing_set.load(self.folder["testing_set"], "testing_set")
            except OSError:
                return original_initialize_testing_set(self, mu_range, ntest, enable_import, sampling, **kwargs)
            else:
                return True
        rbnics.reduction_methods.base.ReductionMethod.initialize_testing_set = initialize_testing_set
        
    original__time_dependent_eim__initialize_training_set = rbnics.eim.reduction_methods.TimeDependentEIMApproximationReductionMethod.initialize_training_set
    def time_dependent_eim__initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
        import_successful = original__time_dependent_eim__initialize_training_set(self, ntrain, True, sampling, **kwargs)
        assert import_successful
        return import_successful
    rbnics.eim.reduction_methods.TimeDependentEIMApproximationReductionMethod.initialize_training_set = time_dependent_eim__initialize_training_set

    original__time_dependent_eim__initialize_testing_set = rbnics.eim.reduction_methods.TimeDependentEIMApproximationReductionMethod.initialize_testing_set
    def time_dependent_eim__initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
        import_successful = original__time_dependent_eim__initialize_testing_set(self, ntest, True, sampling, **kwargs)
        assert import_successful
        return import_successful
    rbnics.eim.reduction_methods.TimeDependentEIMApproximationReductionMethod.initialize_testing_set = time_dependent_eim__initialize_testing_set
