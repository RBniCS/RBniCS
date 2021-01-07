# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import sys
from logging import basicConfig, DEBUG
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import (
    logger as pull_back_to_reference_domain_logger)
from rbnics.shape_parametrization.problems.affine_shape_parametrization_decorated_problem import (
    logger as affine_shape_parametrization_logger)


def EnableDebug():
    basicConfig(stream=sys.stdout, format="%(message)s")
    affine_shape_parametrization_logger.setLevel(DEBUG)
    pull_back_to_reference_domain_logger.setLevel(DEBUG)

    def EnableDebug_Decorator(ParametrizedDifferentialProblem_DerivedClass):
        # return value (a class) for the decorator
        return ParametrizedDifferentialProblem_DerivedClass

    # return the decorator itself
    return EnableDebug_Decorator
