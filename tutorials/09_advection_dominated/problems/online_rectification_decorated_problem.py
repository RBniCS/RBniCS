# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ProblemDecoratorFor


def OnlineRectificationDecoratedProblem(**decorator_kwargs):
    from .online_rectification import OnlineRectification

    @ProblemDecoratorFor(OnlineRectification)
    def OnlineRectificationDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):
        # return value (a class) for the decorator
        return EllipticCoerciveProblem_DerivedClass

    # return the decorator itself
    return OnlineRectificationDecoratedProblem_Decorator
