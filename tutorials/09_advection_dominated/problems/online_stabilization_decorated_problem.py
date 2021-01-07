# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor


def OnlineStabilizationDecoratedProblem(**decorator_kwargs):
    from .online_stabilization import OnlineStabilization

    @ProblemDecoratorFor(OnlineStabilization)
    def OnlineStabilizationDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):

        @PreserveClassName
        class OnlineStabilizationDecoratedProblem_Class(EllipticCoerciveProblem_DerivedClass):

            def __init__(self, V, **kwargs):
                # Flag to enable or disable stabilization
                self.stabilized = True
                # Call to parent
                EllipticCoerciveProblem_DerivedClass.__init__(self, V, **kwargs)

        # return value (a class) for the decorator
        return OnlineStabilizationDecoratedProblem_Class

    # return the decorator itself
    return OnlineStabilizationDecoratedProblem_Decorator
