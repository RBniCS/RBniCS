# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticCoerciveProblem
from rbnics.problems.parabolic.abstract_parabolic_problem import AbstractParabolicProblem

ParabolicCoerciveProblem_Base = AbstractParabolicProblem(EllipticCoerciveProblem)


# Base class containing the definition of parabolic coercive problems
class ParabolicCoerciveProblem(ParabolicCoerciveProblem_Base):
    pass
