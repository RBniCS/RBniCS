# Copyright (C) 2015-2016 by the RBniCS authors
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
__copyright__ = "Copyright 2015-2016 by the RBniCS authors"
__license__ = "LGPL"
__version__ = "0.0.1"
__email__ = "francesco.ballarin@sissa.it, gianluigi.rozza@sissa.it, alberto.sartori@sissa.it"

# Define the __init__.py file. The following instructions are executed
# when the import of RBniCS is done.

# This is the base class, which is inherited by all other classes. It
# defines the base interface with variables and functions that the
# derived classes have to set and/or overwrite.
from parametrized_problem import *

# The first kind of problem to be derived from the class
# ParametrizedProblem is the class EllipticCorciveBase. This class
# defines and implement variables and methods that needed for solving
# an elliptic and coercive problem. This class specializes in the two
# currently implemented reduced order methods, namely the Reduced
# Basis Method (EllipticCoerciveRBBase), and the Proper Orthogonal
# Decomposition (EllipticCoercivePODBase). These two classes assume
# that the output(s) of interest is (are) compliant. Whether the
# compliancy hypothesis does not hold, the
# EllipticCoerciveRBNonCompliantBase must be used.
from elliptic_coercive_base import *

# Compliant case
from elliptic_coercive_rb_base import *
from elliptic_coercive_pod_base import *

# Non compliant case
from elliptic_coercive_rb_non_compliant_base import *

# The common interface for handling time dependent problem is
# implemented in the ParabolicCoerciveBase, which is derived from
# EllipticCoerciveBase. Indeed, at each temporal step, and elliptic
# problem is solved. As for the elliptic case, this class is
# specialized for both the Reduced Basis method as well as for the
# Proper Orthogonal Decomposition. In both case, the compliancy is assumed to hold.
from parabolic_coercive_base import *

# Compliant case
from parabolic_coercive_pod_base import *
from parabolic_coercive_rb_base import *


#### Helper functions and classes

# This class implements the Gram Schmidt orthogonalization
from gram_schmidt import *

# This class provides the methods needed to perform the Proper Orthogonal Decomposition
from proper_orthogonal_decomposition import *

# This class is dedicated to the implementation of the Successive
# Constraint Method (SCM), which is used within the Reduced Basis
# approach in order to have an estimation of the lower bound of the
# coercivty constant. Such lower bound enters the a posteriori error
# estimation.
from scm import *

# If the geometry is parametrized, this class takes care of producing
# nice plots where the mesh is properly deformed.
from shape_parametrization import *

# In this class the Empirical interpolation Method (EIM) is coded,
# which is a standard procedure which leads to an approximate affine
# decomposition when the problem does not admit one.
from eim import *


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

