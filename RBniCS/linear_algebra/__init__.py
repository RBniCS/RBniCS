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
## @file __init__.py
#  @brief Init file for auxiliary linear algebra module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.linear_algebra.affine_expansion_offline_storage import AffineExpansionOfflineStorage
from RBniCS.linear_algebra.affine_expansion_online_storage import AffineExpansionOnlineStorage
from RBniCS.linear_algebra.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.linear_algebra.functions_list import FunctionsList
from RBniCS.linear_algebra.gram_schmidt import GramSchmidt
from RBniCS.linear_algebra.online_eigen_solver import OnlineEigenSolver
from RBniCS.linear_algebra.online_function import OnlineFunction
from RBniCS.linear_algebra.online_matrix import OnlineMatrix
from RBniCS.linear_algebra.online_vector import OnlineVector
from RBniCS.linear_algebra.product import product
from RBniCS.linear_algebra.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.linear_algebra.snapshots_matrix import SnapshotsMatrix
from RBniCS.linear_algebra.solve import solve
from RBniCS.linear_algebra.sum import sum
from RBniCS.linear_algebra.transpose import transpose
from RBniCS.linear_algebra.truth_eigen_solver import TruthEigenSolver
from RBniCS.linear_algebra.truth_function import TruthFunction
from RBniCS.linear_algebra.truth_matrix import TruthMatrix
from RBniCS.linear_algebra.truth_vector import TruthVector

__all__ = [
    'AffineExpansionOfflineStorage',
    'AffineExpansionOnlineStorage',
    'BasisFunctionsMatrix',
    'FunctionsList',
    'GramSchmidt',
    'OnlineEigenSolver',
    'OnlineFunction',
    'OnlineMatrix',
    'OnlineVector',
    'product',
    'ProperOrthogonalDecomposition',
    'SnapshotsMatrix',
    'solve',
    'sum',
    'transpose',
    'TruthEigenSolver',
    'TruthFunction',
    'TruthMatrix',
    'TruthVector'
]
