# Copyright (C) 2015-2017 by the RBniCS authors
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

from RBniCS.backends.numpy.abs import abs
from RBniCS.backends.numpy.adjoint import adjoint
from RBniCS.backends.numpy.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.numpy.assign import assign
from RBniCS.backends.numpy.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.backends.numpy.copy import copy
from RBniCS.backends.numpy.eigen_solver import EigenSolver
from RBniCS.backends.numpy.evaluate import evaluate
from RBniCS.backends.numpy.export import export
from RBniCS.backends.numpy.function import Function
from RBniCS.backends.numpy.functions_list import FunctionsList
from RBniCS.backends.numpy.gram_schmidt import GramSchmidt
from RBniCS.backends.numpy.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from RBniCS.backends.numpy.import_ import import_
from RBniCS.backends.numpy.linear_solver import LinearSolver
from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.backends.numpy.max import max
#from RBniCS.backends.numpy.mesh_motion import MeshMotion
from RBniCS.backends.numpy.nonlinear_solver import NonlinearSolver
#from RBniCS.backends.numpy.parametrized_expression_factory import ParametrizedExpressionFactory
#from RBniCS.backends.numpy.parametrized_tensor_factory import ParametrizedTensorFactory
from RBniCS.backends.numpy.product import product
from RBniCS.backends.numpy.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
#from RBniCS.backends.numpy.reduced_mesh import ReducedMesh
#from RBniCS.backends.numpy.reduced_vertices import ReducedVertices
#from RBniCS.backends.numpy.separated_parametrized_form import SeparatedParametrizedForm
from RBniCS.backends.numpy.snapshots_matrix import SnapshotsMatrix
from RBniCS.backends.numpy.sum import sum
from RBniCS.backends.numpy.tensor_basis_list import TensorBasisList
from RBniCS.backends.numpy.tensor_snapshots_list import TensorSnapshotsList
from RBniCS.backends.numpy.tensors_list import TensorsList
from RBniCS.backends.numpy.time_stepping import TimeStepping
from RBniCS.backends.numpy.transpose import transpose
from RBniCS.backends.numpy.vector import Vector

__all__ = [
    'abs',
    'adjoint',
    'AffineExpansionStorage',
    'assign',
    'BasisFunctionsMatrix',
    'copy',
    'EigenSolver',
    'evaluate',
    'export',
    'Function',
    'FunctionsList',
    'GramSchmidt',
    'HighOrderProperOrthogonalDecomposition',
    'import_',
    'LinearSolver',
    'Matrix',
    'max',
#    'MeshMotion',
    'NonlinearSolver',
#    'ParametrizedExpressionFactory',
#    'ParametrizedTensorFactory',
    'product',
    'ProperOrthogonalDecomposition',
#    'ReducedMesh',
#    'ReducedVertices',
#    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'TensorBasisList',
    'TensorSnapshotsList',
    'TensorsList',
    'TimeStepping',
    'transpose',
    'Vector'
]