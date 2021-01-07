# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor


def OnlineVanishingViscosityDecoratedProblem(viscosity, N_threshold_min, N_threshold_max, **decorator_kwargs):
    from .online_vanishing_viscosity import OnlineVanishingViscosity

    @ProblemDecoratorFor(
        OnlineVanishingViscosity,
        viscosity=viscosity,
        N_threshold_min=N_threshold_min,
        N_threshold_max=N_threshold_max
    )
    def OnlineVanishingViscosityDecoratedProblem_Decorator(EllipticCoerciveProblem_DerivedClass):

        @PreserveClassName
        class OnlineVanishingViscosityDecoratedProblem_Class(EllipticCoerciveProblem_DerivedClass):

            def __init__(self, V, **kwargs):
                # Store input parameters from the decorator factory
                self._viscosity = viscosity
                self._N_threshold_min = N_threshold_min
                self._N_threshold_max = N_threshold_max
                assert self._viscosity >= 0.
                assert self._N_threshold_min >= 0.
                assert self._N_threshold_max >= 0.
                assert self._N_threshold_min <= 1.
                assert self._N_threshold_max <= 1.
                assert self._N_threshold_min < self._N_threshold_max
                # Flag to enable or disable stabilization
                self.stabilized = True
                # Call to parent
                EllipticCoerciveProblem_DerivedClass.__init__(self, V, **kwargs)

        # return value (a class) for the decorator
        return OnlineVanishingViscosityDecoratedProblem_Class

    # return the decorator itself
    return OnlineVanishingViscosityDecoratedProblem_Decorator
