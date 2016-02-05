# Copyright (C) 2015-2016 SISSA mathLab
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#
## RBniCS: reduced order modelling in FEniCS

__author__ = "Francesco Ballarin, Gianluigi Rozza, Alberto Sartori"
__copyright__ = "Copyright 2015-2016 SISSA mathLab"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"


from eim import *
from elliptic_coercive_base import *
from elliptic_coercive_pod_base import *
from elliptic_coercive_rb_base import *
from elliptic_coercive_rb_non_compliant_base import *
from gram_schmidt import *
from parabolic_coercive_base import *
from parabolic_coercive_pod_base import *
from parabolic_coercive_rb_base import *
from parametrized_problem import *
from proper_orthogonal_decomposition import *
from scm import *
from shape_parametrization import *

__all__ = [ \
              'EIM', \
              'EllipticCoerciveBase', \
              'EllipticCoercivePODBase', \
              'EllipticCoerciveRBBase', \
              'EllipticCoerciveRBNonCompliantBase', \
              'GramSchmidt', \
              'ParabolicCoerciveBase', \
              'ParabolicCoercivePODBase', \
              'ParabolicCoerciveRBBase', \
              'ParametrizedProblem', \
              'ProperOrthogonalDecomposition', \
              'SCM', \
              'ShapeParametrization', \
          ]

