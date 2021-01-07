# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.eim.problems.exact_parametrized_functions_decorated_problem import (
    ExactParametrizedFunctionsDecoratedProblem)

# For the sake of the user, since this is the only class that he/she needs to use,
# rename the decorated problem to an easier name
ExactParametrizedFunctions = ExactParametrizedFunctionsDecoratedProblem
