# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators.preserve_class_name import PreserveClassName
from rbnics.utils.decorators.store_problem_decorators_for_factories import StoreProblemDecoratorsForFactories


def ProblemDecoratorFor(Algorithm, ExactAlgorithm=None, replaces=None, replaces_if=None, **kwargs):
    def ProblemDecoratorFor_Decorator(ProblemDecorator):
        def ProblemDecorator_WithStorage(Problem):
            @StoreProblemDecoratorsForFactories(Problem, Algorithm, ExactAlgorithm, **kwargs)
            @PreserveClassName
            class DecoratedProblem(ProblemDecorator(Problem)):
                pass

            # Return
            return DecoratedProblem
        # Done with the storage, return the new problem decorator
        return ProblemDecorator_WithStorage
    return ProblemDecoratorFor_Decorator
