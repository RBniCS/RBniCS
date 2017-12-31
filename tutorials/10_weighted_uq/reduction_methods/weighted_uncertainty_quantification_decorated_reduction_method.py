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

from math import sqrt
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.utils.mpi import log, DEBUG
from problems import WeightedUncertaintyQuantification

@ReductionMethodDecoratorFor(WeightedUncertaintyQuantification)
def WeightedUncertaintyQuantificationDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):
    
    @PreserveClassName
    class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base(EllipticCoerciveReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            EllipticCoerciveReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            self.weight = None
            self.training_set_density = None
            
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, weight=None, **kwargs):
            import_successful = EllipticCoerciveReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            self.weight = weight
            return import_successful
            
        def _offline(self):
            # Initialize densities
            tranining_set_and_first_mu = [mu for mu in self.training_set]
            tranining_set_and_first_mu.append(self.truth_problem.mu)
            if self.weight is not None:
                self.training_set_density = dict(zip(tranining_set_and_first_mu, self.weight.density(self.truth_problem.mu_range, tranining_set_and_first_mu)))
            else:
                self.training_set_density = {mu: 1. for mu in tranining_set_and_first_mu}
            
            # Call Parent method
            EllipticCoerciveReductionMethod_DerivedClass._offline(self)
            
    if hasattr(EllipticCoerciveReductionMethod_DerivedClass, "greedy"): # RB reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def _greedy(self):
                """
                It chooses the next parameter in the offline stage in a greedy fashion. Internal method.
                
                :return: max error estimator and the respective parameter.
                """
                
                def weight(mu):
                    return sqrt(self.training_set_density[mu])
                
                # Print some additional information on the consistency of the reduced basis
                print("absolute error for current mu =", self.reduced_problem.compute_error())
                print("absolute (weighted) error estimator for current mu =", weight(self.truth_problem.mu)*self.reduced_problem.estimate_error())
                print("absolute non-weighted error estimator for current mu =", self.reduced_problem.estimate_error())
                
                # Carry out the actual greedy search
                def solve_and_estimate_error(mu):
                    self.reduced_problem.set_mu(mu)
                    self.reduced_problem.solve()
                    error_estimator = self.reduced_problem.estimate_error()
                    weighted_error_estimator = weight(mu)*error_estimator
                    log(DEBUG, "(Weighted) error estimator for mu = " + str(mu) + " is " + str(weighted_error_estimator))
                    log(DEBUG, "Non-weighted error estimator for mu = " + str(mu) + " is " + str(error_estimator))
                    return weighted_error_estimator
                
                print("find next mu")
                return self.training_set.max(solve_and_estimate_error)
    else: # POD reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def update_snapshots_matrix(self, snapshot):
                def weight(mu):
                    return sqrt(self.training_set_density[mu])
                    
                self.POD.store_snapshot(snapshot, weight=weight(self.truth_problem.mu))
            
    # return value (a class) for the decorator
    return WeightedUncertaintyQuantificationDecoratedReductionMethod_Class
