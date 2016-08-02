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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, ReductionMethodDecoratorFor
from RBniCS.eim.problems import EIM
from RBniCS.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod

@ReductionMethodDecoratorFor(EIM)
def EIMDecoratedReductionMethod(ReductionMethod_DerivedClass):
    
    @Extends(ReductionMethod_DerivedClass, preserve_class_name=True)
    class EIMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            # Storage for EIM reduction methods
            self.EIM_reductions = dict() # from coefficients to _EIMReductionMethod
            
            # Preprocess each term in the affine expansions
            for coeff in self.truth_problem.EIM_approximations:
                self.EIM_reductions[coeff] = EIMApproximationReductionMethod(self.truth_problem.EIM_approximations[coeff], type(self.truth_problem).__name__ + "/eim/" + str(coeff.hash_code))
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the EIM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "EIM" in kwargs
            Nmax_EIM = kwargs["EIM"]
            if isinstance(Nmax_EIM, dict):
                for term in self.separated_forms:
                    for q in range(len(self.separated_forms[term])):
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                assert term in Nmax_EIM and q in Nmax_EIM[term]
                                assert coeff in self.EIM_reductions
                                self.EIM_reductions[coeff].set_Nmax(max(self.EIM_reductions[coeff].Nmax, Nmax_EIM[term][q])) # kwargs are not needed
            else:
                assert isinstance(Nmax_EIM, int)
                for coeff in self.EIM_reductions:
                    self.EIM_reductions[coeff].set_Nmax(Nmax_EIM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            # Since exact evaluation is required, we cannot use a distributed xi_train
            self.xi_train.distributed_max = False
            for coeff in self.EIM_reductions:
                import_successful_EIM = self.EIM_reductions[coeff].set_xi_train(ntrain, enable_import, sampling)
                import_successful = import_successful and import_successful_EIM
            return import_successful
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        @override
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            for coeff in self.EIM_reductions:
                import_successful_EIM = self.EIM_reductions[coeff].set_xi_test(ntest, enable_import, sampling)
                import_successful = import_successful and import_successful_EIM
            return import_successful
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            # Perform first the EIM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            for coeff in self.EIM_reductions:
                self.EIM_reductions[coeff].offline()
            # ..., and then call the parent method.
            self.truth_problem.set_mu(bak_first_mu)
            return ReductionMethod_DerivedClass.offline(self)
    
        #  @}
        ########################### end - OFFLINE STAGE - end ###########################
    
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
    
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the test set
        @override
        def error_analysis(self, N=None, with_respect_to=None, **kwargs):
            # Perform first the EIM error analysis, ...
            if (
                with_respect_to is None # otherwise we assume the user was interested in computing the error w.r.t. an exact parametrized functions, 
                                        # so he probably is not interested in the error analysis of EIM
                    and
                len(kwargs) == 0        # otherwise we assume the user was interested in computing the error for a fixed number of EIM basis
                                        # functions, thus he has already carried out the error analysis of EIM
            ):
                for coeff in self.EIM_reductions:
                    self.EIM_reductions[coeff].error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N, with_respect_to=with_respect_to, **kwargs)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
    
