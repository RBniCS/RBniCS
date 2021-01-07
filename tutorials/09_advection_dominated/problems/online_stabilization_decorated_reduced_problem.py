# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import OrderedDict
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor
from backends.online import OnlineSolveKwargsGenerator
from .online_stabilization import OnlineStabilization


@ReducedProblemDecoratorFor(OnlineStabilization)
def OnlineStabilizationDecoratedReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):

    @PreserveClassName
    class OnlineStabilizationDecoratedReducedProblem_Class(EllipticCoerciveReducedProblem_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call to parent
            EllipticCoerciveReducedProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Default values for keyword arguments in solve
            self._online_solve_default_kwargs = OrderedDict()
            self._online_solve_default_kwargs["online_stabilization"] = True
            self.OnlineSolveKwargs = OnlineSolveKwargsGenerator(**self._online_solve_default_kwargs)

        def _online_size_from_kwargs(self, N, **kwargs):
            N, kwargs = EllipticCoerciveReducedProblem_DerivedClass._online_size_from_kwargs(self, N, **kwargs)
            kwargs = self.OnlineSolveKwargs(**kwargs)
            return N, kwargs

        def _solve(self, N, **kwargs):
            # Temporarily change value of stabilized attribute in truth problem
            bak_stabilized = self.truth_problem.stabilized
            self.truth_problem.stabilized = kwargs["online_stabilization"]
            # Solve reduced problem
            EllipticCoerciveReducedProblem_DerivedClass._solve(self, N, **kwargs)
            # Restore original value of stabilized attribute in truth problem
            self.truth_problem.stabilized = bak_stabilized

    # return value (a class) for the decorator
    return OnlineStabilizationDecoratedReducedProblem_Class
