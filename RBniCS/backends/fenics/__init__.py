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

# Check that dolfin has been compiled with PETSc and SLEPc
from dolfin import has_petsc, has_linear_algebra_backend, parameters, has_slepc
assert has_petsc() 
assert has_linear_algebra_backend("PETSc") 
assert parameters.linear_algebra_backend == "PETSc"
assert has_slepc()

# Import modules
from RBniCS.backends.fenics.abs import abs
from RBniCS.backends.fenics.affine_expansion_storage import AffineExpansionStorage
from RBniCS.backends.fenics.assign import assign
from RBniCS.backends.fenics.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.backends.fenics.difference import difference
from RBniCS.backends.fenics.eigen_solver import EigenSolver
from RBniCS.backends.fenics.evaluate import evaluate
from RBniCS.backends.fenics.export import export
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.gram_schmidt import GramSchmidt
from RBniCS.backends.fenics.high_order_proper_orthogonal_decomposition import HighOrderProperOrthogonalDecomposition
from RBniCS.backends.fenics.linear_solver import LinearSolver
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.max import max
from RBniCS.backends.fenics.mesh_motion import MeshMotion
from RBniCS.backends.fenics.nonlinear_solver import NonlinearSolver
from RBniCS.backends.fenics.parametrized_expression_factory import ParametrizedExpressionFactory
from RBniCS.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from RBniCS.backends.fenics.product import product
from RBniCS.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.fenics.rescale import rescale
from RBniCS.backends.fenics.separated_parametrized_form import SeparatedParametrizedForm
from RBniCS.backends.fenics.snapshots_matrix import SnapshotsMatrix
from RBniCS.backends.fenics.sum import sum
from RBniCS.backends.fenics.tensor_basis_list import TensorBasisList
from RBniCS.backends.fenics.tensor_snapshots_list import TensorSnapshotsList
from RBniCS.backends.fenics.tensors_list import TensorsList
from RBniCS.backends.fenics.time_stepping import TimeStepping
from RBniCS.backends.fenics.transpose import transpose
from RBniCS.backends.fenics.vector import Vector

__all__ = [
    'abs',
    'AffineExpansionStorage',
    'assign',
    'BasisFunctionsMatrix',
    'difference',
    'EigenSolver',
    'evaluate',
    'export',
    'Function',
    'FunctionsList',
    'GramSchmidt',
    'HighOrderProperOrthogonalDecomposition',
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
    'rescale',
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
