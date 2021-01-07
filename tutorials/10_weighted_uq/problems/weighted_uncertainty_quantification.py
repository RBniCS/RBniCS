# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .weighted_uncertainty_quantification_decorated_problem import WeightedUncertaintyQuantificationDecoratedProblem

# For the sake of the user, since this is the only class that he/she needs to use,
# rename the decorated problem to an easier name
WeightedUncertaintyQuantification = WeightedUncertaintyQuantificationDecoratedProblem
