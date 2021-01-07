# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from rbnics.utils.decorators import ReductionMethodFor
from problems import GeostrophicProblem
from .geostrophic_reduction_method import GeostrophicReductionMethod

GeostrophicPODGalerkinReduction_Base = LinearPODGalerkinReduction(
    GeostrophicReductionMethod(DifferentialProblemReductionMethod))


@ReductionMethodFor(GeostrophicProblem, "PODGalerkin")
class GeostrophicPODGalerkinReduction(GeostrophicPODGalerkinReduction_Base):
    pass
