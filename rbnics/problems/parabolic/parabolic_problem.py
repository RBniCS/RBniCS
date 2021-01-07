# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticProblem
from rbnics.problems.parabolic.abstract_parabolic_problem import AbstractParabolicProblem

ParabolicProblem_Base = AbstractParabolicProblem(EllipticProblem)


# Base class containing the definition of parabolic problems
class ParabolicProblem(ParabolicProblem_Base):
    pass
