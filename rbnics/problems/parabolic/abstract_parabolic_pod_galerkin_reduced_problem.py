# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearTimeDependentPODGalerkinReducedProblem


def AbstractParabolicPODGalerkinReducedProblem(AbstractParabolicReducedProblem_DerivedClass):
    AbstractParabolicPODGalerkinReducedProblem_Base = LinearTimeDependentPODGalerkinReducedProblem(
        AbstractParabolicReducedProblem_DerivedClass)

    class AbstractParabolicPODGalerkinReducedProblem_Class(AbstractParabolicPODGalerkinReducedProblem_Base):
        pass

    # return value (a class) for the decorator
    return AbstractParabolicPODGalerkinReducedProblem_Class
