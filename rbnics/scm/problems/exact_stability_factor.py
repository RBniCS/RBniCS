# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.scm.problems.exact_stability_factor_decorated_problem import ExactStabilityFactorDecoratedProblem

# For the sake of the user, since this is the only class that he/she needs to use,
# rename the decorated problem to an easier name
ExactStabilityFactor = ExactStabilityFactorDecoratedProblem
