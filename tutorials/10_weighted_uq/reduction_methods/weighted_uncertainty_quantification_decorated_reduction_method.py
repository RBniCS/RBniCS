# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from math import sqrt
from logging import DEBUG, getLogger
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from problems import WeightedUncertaintyQuantification

logger = getLogger("tutorials/10_weighted_uq/reduction_methods"
                   + "/weighted_uncertainty_quantification_decorated_reduction_method.py")


@ReductionMethodDecoratorFor(WeightedUncertaintyQuantification)
def WeightedUncertaintyQuantificationDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):

    @PreserveClassName
    class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base(
            EllipticCoerciveReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            EllipticCoerciveReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            self.weight = None
            self.training_set_density = None

        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, weight=None, **kwargs):
            import_successful = EllipticCoerciveReductionMethod_DerivedClass.initialize_training_set(
                self, ntrain, enable_import, sampling, **kwargs)
            self.weight = weight
            return import_successful

        def _offline(self):
            # Initialize densities
            tranining_set_and_first_mu = [mu for mu in self.training_set]
            tranining_set_and_first_mu.append(self.truth_problem.mu)
            if self.weight is not None:
                self.training_set_density = dict(zip(
                    tranining_set_and_first_mu, self.weight.density(
                        self.truth_problem.mu_range, tranining_set_and_first_mu)))
            else:
                self.training_set_density = {mu: 1. for mu in tranining_set_and_first_mu}

            # Call Parent method
            EllipticCoerciveReductionMethod_DerivedClass._offline(self)

    if hasattr(EllipticCoerciveReductionMethod_DerivedClass, "greedy"):  # RB reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(
                WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def _greedy(self):
                """
                It chooses the next parameter in the offline stage in a greedy fashion. Internal method.

                :return: max error estimator and the respective parameter.
                """

                def weight(mu):
                    return sqrt(self.training_set_density[mu])

                if self.reduced_problem.N > 0:  # skip during initialization
                    # Print some additional information on the consistency of the reduced basis
                    print("absolute error for current mu =", self.reduced_problem.compute_error())
                    print("absolute (weighted) error estimator for current mu =",
                          weight(self.truth_problem.mu) * self.reduced_problem.estimate_error())
                    print("absolute non-weighted error estimator for current mu =",
                          self.reduced_problem.estimate_error())

                # Carry out the actual greedy search
                def solve_and_estimate_error(mu):
                    self.reduced_problem.set_mu(mu)
                    self.reduced_problem.solve()
                    error_estimator = self.reduced_problem.estimate_error()
                    weighted_error_estimator = weight(mu) * error_estimator
                    logger.log(DEBUG, "(Weighted) error estimator for mu = " + str(mu) + " is "
                               + str(weighted_error_estimator))
                    logger.log(DEBUG, "Non-weighted error estimator for mu = " + str(mu) + " is "
                               + str(error_estimator))
                    return weighted_error_estimator

                print("find next mu")
                return self.training_set.max(solve_and_estimate_error)
    else:  # POD reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(
                WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def update_snapshots_matrix(self, snapshot):
                def weight(mu):
                    return sqrt(self.training_set_density[mu])

                self.POD.store_snapshot(snapshot, weight=weight(self.truth_problem.mu))

    # return value (a class) for the decorator
    return WeightedUncertaintyQuantificationDecoratedReductionMethod_Class
