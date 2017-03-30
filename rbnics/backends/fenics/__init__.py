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

# Check that dolfin has been compiled with PETSc and SLEPc
from dolfin import has_petsc, has_linear_algebra_backend, parameters, has_slepc
assert has_petsc() 
assert has_linear_algebra_backend("PETSc") 
assert parameters.linear_algebra_backend == "PETSc"
assert has_slepc()

# Import modules
from rbnics.backends.fenics.abs import abs
from rbnics.backends.fenics.adjoint import adjoint
from rbnics.backends.fenics.affine_expansion_storage import AffineExpansionStorage
from rbnics.backends.fenics.assign import assign
from rbnics.backends.fenics.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.fenics.copy import copy
from rbnics.backends.fenics.eigen_solver import EigenSolver
from rbnics.backends.fenics.evaluate import evaluate
from rbnics.backends.fenics.export import export
from rbnics.backends.fenics.function import Function
from rbnics.backends.fenics.functions_list import FunctionsList
from rbnics.backends.fenics.gram_schmidt import GramSchmidt
from rbnics.backends.fenics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from rbnics.backends.fenics.import_ import import_
from rbnics.backends.fenics.linear_solver import LinearSolver
from rbnics.backends.fenics.matrix import Matrix
from rbnics.backends.fenics.max import max
from rbnics.backends.fenics.mesh_motion import MeshMotion
from rbnics.backends.fenics.nonlinear_solver import NonlinearSolver
from rbnics.backends.fenics.parametrized_expression_factory import ParametrizedExpressionFactory
from rbnics.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.fenics.product import product
from rbnics.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from rbnics.backends.fenics.reduced_mesh import ReducedMesh
from rbnics.backends.fenics.reduced_vertices import ReducedVertices
from rbnics.backends.fenics.separated_parametrized_form import SeparatedParametrizedForm
from rbnics.backends.fenics.snapshots_matrix import SnapshotsMatrix
from rbnics.backends.fenics.sum import sum
from rbnics.backends.fenics.tensor_basis_list import TensorBasisList
from rbnics.backends.fenics.tensor_snapshots_list import TensorSnapshotsList
from rbnics.backends.fenics.tensors_list import TensorsList
from rbnics.backends.fenics.time_stepping import TimeStepping
from rbnics.backends.fenics.transpose import transpose
from rbnics.backends.fenics.vector import Vector

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
    'MeshMotion',
    'NonlinearSolver',
    'ParametrizedExpressionFactory',
    'ParametrizedTensorFactory',
    'product',
    'ProperOrthogonalDecomposition',
    'ReducedMesh',
    'ReducedVertices',
    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'TensorBasisList',
    'TensorSnapshotsList',
    'TensorsList',
    'TimeStepping',
    'transpose',
    'Vector'
]
