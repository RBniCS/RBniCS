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

from RBniCS.backends.fenics.wrapping_utils.create_submesh import create_submesh, create_submesh_subdomains, mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs
from RBniCS.backends.fenics.wrapping_utils.dirichlet_bc import DirichletBC
from RBniCS.backends.fenics.wrapping_utils.dofs_parallel_io_helpers import build_dof_map_writer_mapping, build_dof_map_reader_mapping
from RBniCS.backends.fenics.wrapping_utils.function_from_ufl_operators import function_from_ufl_operators
from RBniCS.backends.fenics.wrapping_utils.function_space import FunctionSpace
from RBniCS.backends.fenics.wrapping_utils.get_expression_description import get_expression_description
from RBniCS.backends.fenics.wrapping_utils.get_form_argument import get_form_argument
from RBniCS.backends.fenics.wrapping_utils.get_form_description import get_form_description
from RBniCS.backends.fenics.wrapping_utils.get_form_name import get_form_name
from RBniCS.backends.fenics.wrapping_utils.parametrized_constant import ParametrizedConstant
from RBniCS.backends.fenics.wrapping_utils.parametrized_expression import ParametrizedExpression
from RBniCS.backends.fenics.wrapping_utils.plot import plot

__all__ = [
    'build_dof_map_reader_mapping',
    'build_dof_map_writer_mapping',
    'create_submesh',
    'create_submesh_subdomains',
    'DirichletBC',
    'function_from_ufl_operators',
    'FunctionSpace',
    'get_expression_description',
    'get_form_argument',
    'get_form_description',
    'get_form_name',
    'mesh_dofs_to_submesh_dofs',
    'ParametrizedConstant',
    'ParametrizedExpression',
    'plot',
    'submesh_dofs_to_mesh_dofs'
]

__overridden__ = [
    'DirichletBC',
    'FunctionSpace',
    'ParametrizedExpression',
    'plot'
]
