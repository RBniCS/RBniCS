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
from RBniCS.eim.problems import DEIM
from RBniCS.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod as DEIMApproximationReductionMethod

@ReductionMethodDecoratorFor(DEIM)
def DEIMDecoratedReductionMethod(ReductionMethod_DerivedClass):
    
    @Extends(ReductionMethod_DerivedClass, preserve_class_name=True)
    class DEIMDecoratedReductionMethod_Class(ReductionMethod_DerivedClass):
        @override
        def __init__(self, truth_problem):
            # Call the parent initialization
            ReductionMethod_DerivedClass.__init__(self, truth_problem)
            # Storage for DEIM reduction methods
            self.DEIM_reductions = dict() # from term to dict of DEIMApproximationReductionMethod
            
            # Preprocess each term in the affine expansions
            for (term, DEIM_approximations_term) in self.truth_problem.DEIM_approximations.iteritems():
                self.DEIM_reductions[term] = dict()
                for (q, DEIM_approximations_term_q) in DEIM_approximations_term.iteritems():
                    self.DEIM_reductions[term][q] = DEIMApproximationReductionMethod(DEIM_approximations_term_q)
            
        ###########################     SETTERS     ########################### 
        ## @defgroup Setters Set properties of the reduced order approximation
        #  @{
    
        # Propagate the values of all setters also to the DEIM object
        
        ## OFFLINE: set maximum reduced space dimension (stopping criterion)
        @override
        def set_Nmax(self, Nmax, **kwargs):
            ReductionMethod_DerivedClass.set_Nmax(self, Nmax, **kwargs)
            assert "DEIM" in kwargs
            Nmax_DEIM = kwargs["DEIM"]
            if isinstance(Nmax_DEIM, dict):
                for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                    assert term in Nmax_DEIM
                    assert len(self.DEIM_reductions[term]) == len(Nmax_DEIM[term])
                    for (q, Nmax_DEIM_term_q) in Nmax_DEIM[term].iteritems():
                        DEIM_reductions_term[q].set_Nmax(Nmax_DEIM_term_q) # kwargs are not needed
            else:
                assert isinstance(Nmax_DEIM, int)
                for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                    for (_, DEIM_reduction_term_q) in DEIM_reductions_term.iteritems():
                        DEIM_reduction_term_q.set_Nmax(Nmax_DEIM) # kwargs are not needed

            
        ## OFFLINE: set the elements in the training set \xi_train.
        @override
        def set_xi_train(self, ntrain, enable_import=True, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_train(self, ntrain, enable_import, sampling)
            # Since exact evaluation is required, we cannot use a distributed xi_train
            self.xi_train.distributed_max = False
            for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                for (_, DEIM_reduction_term_q) in DEIM_reductions_term.iteritems():
                    import_successful_DEIM = DEIM_reduction_term_q.set_xi_train(ntrain, enable_import, sampling)
                    import_successful = import_successful and import_successful_DEIM
            return import_successful
            
        ## ERROR ANALYSIS: set the elements in the test set \xi_test.
        @override
        def set_xi_test(self, ntest, enable_import=False, sampling=None):
            import_successful = ReductionMethod_DerivedClass.set_xi_test(self, ntest, enable_import, sampling)
            for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                for (_, DEIM_reduction_term_q) in DEIM_reductions_term.iteritems():
                    import_successful_DEIM = DEIM_reduction_term_q.set_xi_test(ntest, enable_import, sampling)
                    import_successful = import_successful and import_successful_DEIM
            return import_successful
            
        #  @}
        ########################### end - SETTERS - end ########################### 
        
        ###########################     OFFLINE STAGE     ########################### 
        ## @defgroup OfflineStage Methods related to the offline stage
        #  @{
    
        ## Perform the offline phase of the reduced order model
        @override
        def offline(self):
            # Perform first the DEIM offline phase, ...
            bak_first_mu = tuple(list(self.truth_problem.mu))
            for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                for (_, DEIM_reduction_term_q) in DEIM_reductions_term.iteritems():
                    DEIM_reduction_term_q.offline()
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
        def error_analysis(self, N=None, **kwargs):
            # Perform first the DEIM error analysis, ...
            if (
                "with_respect_to" not in kwargs # otherwise we assume the user was interested in computing the error w.r.t. 
                                                # an exact parametrized functions, 
                                                # so he probably is not interested in the error analysis of DEIM
                    and
                "N_DEIM" not in kwargs           # otherwise we assume the user was interested in computing the error for a fixed number of DEIM basis
                                                # functions, thus he has already carried out the error analysis of DEIM
            ):
                for (term, DEIM_reductions_term) in self.DEIM_reductions.iteritems():
                    for (_, DEIM_reduction_term_q) in DEIM_reductions_term.iteritems():
                        DEIM_reduction_term_q.error_analysis(N)
            # ..., and then call the parent method.
            ReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
            
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return DEIMDecoratedReductionMethod_Class
    
