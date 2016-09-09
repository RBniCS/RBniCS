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

from RBniCS.backends.abstract.abs import abs
from RBniCS.backends.abstract.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.abstract.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.backends.abstract.difference import difference
from RBniCS.backends.abstract.evaluate import evaluate
from RBniCS.backends.abstract.eigen_solver import EigenSolver
from RBniCS.backends.abstract.function import Function
from RBniCS.backends.abstract.functions_list import FunctionsList
from RBniCS.backends.abstract.gram_schmidt import GramSchmidt
from RBniCS.backends.abstract.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from RBniCS.backends.abstract.linear_solver import LinearSolver
from RBniCS.backends.abstract.matrix import Matrix
from RBniCS.backends.abstract.max import max
from RBniCS.backends.abstract.product import product
from RBniCS.backends.abstract.projected_parametrized_expression import ProjectedParametrizedExpression
from RBniCS.backends.abstract.projected_parametrized_tensor import ProjectedParametrizedTensor
from RBniCS.backends.abstract.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.abstract.reduced_mesh import ReducedMesh
from RBniCS.backends.abstract.reduced_vertices import ReducedVertices
from RBniCS.backends.abstract.rescale import rescale
from RBniCS.backends.abstract.separated_parametrized_form import SeparatedParametrizedForm
from RBniCS.backends.abstract.snapshots_matrix import SnapshotsMatrix
from RBniCS.backends.abstract.sum import sum
from RBniCS.backends.abstract.tensor_basis_list import TensorBasisList
from RBniCS.backends.abstract.tensor_snapshots_list import TensorSnapshotsList
from RBniCS.backends.abstract.tensors_list import TensorsList
from RBniCS.backends.abstract.transpose import transpose
from RBniCS.backends.abstract.vector import Vector

__all__ = [
    'abs',
    'AffineExpansionStorage',
    'BasisFunctionsMatrix',
    'difference',
    'evaluate',
    'EigenSolver',
    'Function',
    'FunctionsList',
    'GramSchmidt',
    'HighOrderProperOrthogonalDecomposition',
    'LinearSolver',
    'Matrix',
    'max',
    'product',
    'ProjectedParametrizedExpression',
    'ProjectedParametrizedTensor',
    'ProperOrthogonalDecomposition',
    'ReducedMesh',
    'ReducedVertices',
    'rescale',
    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'TensorBasisList',
    'TensorSnapshotsList',
    'TensorsList',
    'transpose',
    'Vector'
]
