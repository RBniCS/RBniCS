# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.utils.decorators import ReducedProblemFor
from .geostrophic_reduced_problem import GeostrophicReducedProblem
from .geostrophic_problem import GeostrophicProblem
from reduction_methods import GeostrophicPODGalerkinReduction

GeostrophicPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    GeostrophicReducedProblem(ParametrizedReducedDifferentialProblem))


@ReducedProblemFor(GeostrophicProblem, GeostrophicPODGalerkinReduction)
class GeostrophicPODGalerkinReducedProblem(GeostrophicPODGalerkinReducedProblem_Base):
    pass
