# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from rbnics.scm.problems import ExactStabilityFactor


@ReductionMethodDecoratorFor(ExactStabilityFactor)
def ExactStabilityFactorDecoratedReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class ExactStabilityFactorDecoratedReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            DifferentialProblemReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)

        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            import_successful = DifferentialProblemReductionMethod_DerivedClass.initialize_training_set(
                self, ntrain, enable_import, sampling, **kwargs)
            # Since exact evaluation is required, we cannot use a distributed training set
            self.training_set.serialize_maximum_computations()
            return import_successful

    # return value (a class) for the decorator
    return ExactStabilityFactorDecoratedReductionMethod_Class
