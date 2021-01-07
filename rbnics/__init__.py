# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Import system libraries
import os
import sys
import importlib

# Set empty __all__ variable to be possibly extended by backends (implicitly imported below)
__all__ = []

# Process configuration files first
from rbnics.utils.config import config

# Import the minimum subset of RBniCS required to run tutorials
from rbnics.eim.problems import DEIM, EIM, ExactParametrizedFunctions
from rbnics.problems.elliptic import EllipticCoerciveCompliantProblem, EllipticCoerciveProblem, EllipticProblem
from rbnics.problems.elliptic_optimal_control import EllipticOptimalControlProblem
from rbnics.problems.navier_stokes import NavierStokesProblem
from rbnics.problems.navier_stokes_unsteady import NavierStokesUnsteadyProblem
from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
from rbnics.problems.nonlinear_parabolic import NonlinearParabolicProblem
from rbnics.problems.parabolic import ParabolicCoerciveProblem, ParabolicProblem
from rbnics.problems.stokes import StokesProblem
from rbnics.problems.stokes_optimal_control import StokesOptimalControlProblem
from rbnics.problems.stokes_unsteady import StokesUnsteadyProblem
from rbnics.sampling.distributions import (DrawFrom, EquispacedDistribution, LogEquispacedDistribution,
                                           LogUniformDistribution, UniformDistribution)
from rbnics.scm.problems import ExactStabilityFactor, SCM
from rbnics.shape_parametrization.problems import AffineShapeParametrization, ShapeParametrization
from rbnics.utils.decorators import CustomizeReducedProblemFor, CustomizeReductionMethodFor, exact_problem
from rbnics.utils.factories import ReducedBasis, PODGalerkin

__all__ += [
    # rbnics.eim
    "DEIM",
    "EIM",
    "ExactParametrizedFunctions",
    # rbnics.problems
    "EllipticCoerciveCompliantProblem",
    "EllipticCoerciveProblem",
    "EllipticOptimalControlProblem",
    "EllipticProblem",
    "NavierStokesProblem",
    "NavierStokesUnsteadyProblem",
    "NonlinearEllipticProblem",
    "NonlinearParabolicProblem",
    "ParabolicCoerciveProblem",
    "ParabolicProblem",
    "StokesProblem",
    "StokesOptimalControlProblem",
    "StokesUnsteadyProblem",
    # rbnics.sampling
    "DrawFrom",
    "EquispacedDistribution",
    "LogEquispacedDistribution",
    "LogUniformDistribution",
    "UniformDistribution",
    # rbnics.scm
    "ExactStabilityFactor",
    "SCM",
    # rbnics.shape_parametrization
    "AffineShapeParametrization",
    "ShapeParametrization",
    # rbnics.utils.config
    "config",
    # rbnics.utils.decorators
    "CustomizeReducedProblemFor",
    "CustomizeReductionMethodFor",
    "exact_problem",
    # rbnics.utils.factories
    "ReducedBasis",
    "PODGalerkin",
]

# Import remaining modules
rbnics_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
already_imported = ["backends", "eim", "problems", "__pycache__", "reduction_methods", "sampling", "scm",
                    "shape_parametrization", "utils"]
for root, dirs, files in os.walk(os.path.join(rbnics_directory)):
    for dir_ in dirs:
        if dir_ not in already_imported and not dir_.startswith("."):
            importlib.import_module(__name__ + "." + dir_)
            already_imported.append(dir_)
            for class_or_function_name in sys.modules[__name__ + "." + dir_].__all__:
                assert not hasattr(sys.modules[__name__], class_or_function_name)
                setattr(sys.modules[__name__], class_or_function_name, getattr(sys.modules[__name__ + "." + dir_],
                        class_or_function_name))
                sys.modules[__name__].__all__.append(class_or_function_name)
    break  # prevent recursive exploration
