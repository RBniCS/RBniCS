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
def EIMDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class EIMDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Storage for EIM reduction methods
            self.EIM_reductions = dict() # from coefficients to _EIMReductionMethod
            
            # Preprocess each term in the affine expansions
            for coeff in self.truth_problem.EIM_approximations:
                self.EIM_reductions[coeff] = EIMApproximationReductionMethod(self.truth_problem.EIM_approximations[coeff])
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the EIM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            DifferentialProblemReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            # Set Nmax of EIM reductions
            def setter(EIM_reduction, Nmax_EIM):
                EIM_reduction.set_Nmax(max(EIM_reduction.Nmax, Nmax_EIM)) # kwargs are not needed
            self._propagate_setter_from_kwargs_to_EIM_reductions(setter, **kwargs)

            
        ## OFFLINE: set the elements in the training set.
        @override
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            # Since exact evaluation is required, we cannot use a distributed training set
            self.training_set.distributed_max = False
            # Initialize training set of EIM reductions
            def setter(EIM_reduction, ntrain_EIM):
                return EIM_reduction.initialize_training_set(ntrain_EIM, enable_import, sampling) # kwargs are not needed
            import_successful_EIM = self._propagate_setter_from_kwargs_to_EIM_reductions(setter, **kwargs)
            return import_successful and import_successful_EIM
            
        ## ERROR ANALYSIS: set the elements in the testing set.
        @override
        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
            # Initialize testing set of EIM reductions
            def setter(EIM_reduction, ntest_EIM):
                return EIM_reduction.initialize_testing_set(ntest_EIM, enable_import, sampling) # kwargs are not needed
            import_successful_EIM = self._propagate_setter_from_kwargs_to_EIM_reductions(setter, **kwargs)
            return import_successful and import_successful_EIM
            
        def _propagate_setter_from_kwargs_to_EIM_reductions(self, setter, **kwargs):
            assert "EIM" in kwargs
            kwarg_EIM = kwargs["EIM"]
            return_value = True # will be either a bool or None
            if isinstance(kwarg_EIM, dict):
                for term in self.truth_problem.separated_forms:
                    if sum([len(form.coefficients) for form in self.truth_problem.separated_forms[term]]) > 0:
                        assert term in kwarg_EIM, "Please provide a value for term " + str(term)
                        assert isinstance(kwarg_EIM[term], (int, tuple))
                        if isinstance(kwarg_EIM[term], int):
                            kwarg_EIM[term] = [kwarg_EIM[term]]*len(self.truth_problem.separated_forms[term])
                        else:
                            assert len(self.truth_problem.separated_forms[term]) == len(kwarg_EIM[term])
                        for (form, kwarg_EIM_form) in zip(self.truth_problem.separated_forms[term], kwarg_EIM[term]):
                            for addend in form.coefficients:
                                for factor in addend:
                                    assert factor in self.EIM_reductions
                                    current_return_value = setter(self.EIM_reductions[factor], kwarg_EIM_form)
                                    return_value = current_return_value and return_value
            else:
                assert isinstance(kwarg_EIM, int)
                for coeff in self.EIM_reductions:
                    current_return_value = setter(self.EIM_reductions[coeff], kwarg_EIM)
                    return_value = current_return_value and return_value
            return return_value # an "and" with a None results in None, so this method returns only if necessary
            
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
            return DifferentialProblemReductionMethod_DerivedClass.offline(self)
    
        #  @}
        ########################### end - OFFLINE STAGE - end ###########################
    
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
    
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def error_analysis(self, N=None, **kwargs):
            # Perform first the EIM error analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the error w.r.t. 
                                                # an exact parametrized functions, 
                                                # so he probably is not interested in the error analysis of EIM
                    and
                "N_EIM" not in kwargs           # otherwise we assume the user was interested in computing the error for a fixed number of EIM basis
                                                # functions, thus he has already carried out the error analysis of EIM
            ):
                for coeff in self.EIM_reductions:
                    self.EIM_reductions[coeff].error_analysis(N)
            # ..., and then call the parent method.
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return EIMDecoratedReductionMethod_Class
    
