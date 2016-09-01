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
from RBniCS.backends.fenics.basis_functions_matrix import BasisFunctionsMatrix
from RBniCS.backends.fenics.difference import difference
from RBniCS.backends.fenics.evaluate import evaluate
from RBniCS.backends.fenics.eigen_solver import EigenSolver
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.gram_schmidt import GramSchmidt
from RBniCS.backends.fenics.linear_solver import LinearSolver
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.max import max
from RBniCS.backends.fenics.parametrized_matrix import ParametrizedMatrix
from RBniCS.backends.fenics.parametrized_vector import ParametrizedVector
from RBniCS.backends.fenics.product import product
from RBniCS.backends.fenics.projected_parametrized_expression import ProjectedParametrizedExpression
from RBniCS.backends.fenics.proper_orthogonal_decomposition import ProperOrthogonalDecomposition
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.rescale import rescale
from RBniCS.backends.fenics.separated_parametrized_form import SeparatedParametrizedForm
from RBniCS.backends.fenics.snapshots_matrix import SnapshotsMatrix
from RBniCS.backends.fenics.sum import sum
from RBniCS.backends.fenics.transpose import transpose
from RBniCS.backends.fenics.vector import Vector

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
    'LinearSolver',
    'Matrix',
    'max',
    'ParametrizedMatrix',
    'ParametrizedVector',
    'product',
    'projected_expression',
    'ProperOrthogonalDecomposition',
    'ReducedMesh',
    'rescale',
    'SeparatedParametrizedForm',
    'SnapshotsMatrix',
    'sum',
    'transpose',
    'Vector'
]
