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

from mpi4py.MPI import MAX
from dolfin import assemble, inner, dP, TestFunction

def ufl_lagrange_interpolation(output, ufl_expression):
    V = output.function_space()
    if V not in ufl_lagrange_interpolation._test_function:
        ufl_lagrange_interpolation._test_function[V] = TestFunction(V)
    v = ufl_lagrange_interpolation._test_function[V]
    assemble(inner(v, ufl_expression)*dP, output.vector())
ufl_lagrange_interpolation._test_function = dict()
    
def get_global_dof_coordinates(global_dof, V, global_to_local=None, local_dof_to_coordinates=None):
    if global_to_local is None:
        global_to_local = _get_global_dof_to_local_dof_map(V, V.dofmap())
    if local_dof_to_coordinates is None:
        local_dof_to_coordinates = _get_local_dof_to_coordinates_map(V)
    
    mpi_comm = V.mesh().mpi_comm().tompi4py()
    dof_coordinates = None
    dof_coordinates_processor = -1
    if global_dof in global_to_local:
        dof_coordinates = local_dof_to_coordinates[global_to_local[global_dof]]
        dof_coordinates_processor = mpi_comm.rank
    dof_coordinates_processor = mpi_comm.allreduce(dof_coordinates_processor, op=MAX)
    assert dof_coordinates_processor >= 0
    return mpi_comm.bcast(dof_coordinates, root=dof_coordinates_processor)
    
def get_global_dof_component(global_dof, V, global_to_local=None, local_dof_to_component=None):
    if global_to_local is None:
        global_to_local = _get_global_dof_to_local_dof_map(V, V.dofmap())
    if local_dof_to_component is None:
        local_dof_to_component = _get_local_dof_to_component_map(V)
    
    mpi_comm = V.mesh().mpi_comm().tompi4py()
    dof_component = None
    dof_component_processor = -1
    if global_dof in global_to_local:
        dof_component = local_dof_to_component[global_to_local[global_dof]]
        dof_component_processor = mpi_comm.rank
    dof_component_processor = mpi_comm.allreduce(dof_component_processor, op=MAX)
    assert dof_component_processor >= 0
    return mpi_comm.bcast(dof_component, root=dof_component_processor)
    
def assert_lagrange_1(space):
    assert space.ufl_element().family() == "Lagrange", "The current implementation of evaluate relies on CG1 space"
    assert space.ufl_element().degree() == 1, "The current implementation of evaluate relies on CG1 space"
        
# Auxiliary functions:

def _get_global_dof_to_local_dof_map(V, dofmap):
    if V not in _get_global_dof_to_local_dof_map._storage:
        local_to_global = dofmap.tabulate_local_to_global_dofs()
        local_size = dofmap.ownership_range()[1] - dofmap.ownership_range()[0]
        global_to_local = {global_: local for (local, global_) in enumerate(local_to_global) if local < local_size}
        _get_global_dof_to_local_dof_map._storage[V] = global_to_local
    return _get_global_dof_to_local_dof_map._storage[V]
_get_global_dof_to_local_dof_map._storage = dict()
    
def _get_local_dof_to_coordinates_map(V):
    if V not in _get_local_dof_to_coordinates_map._storage:
        _get_local_dof_to_coordinates_map._storage[V] = V.tabulate_dof_coordinates().reshape((-1, V.mesh().ufl_cell().topological_dimension()))
    return _get_local_dof_to_coordinates_map._storage[V]
_get_local_dof_to_coordinates_map._storage = dict()

def _get_local_dof_to_component_map(V, component=None, dof_component_map=None, recursive=False):
    if V not in _get_local_dof_to_component_map._storage:
        if component is None:
            component = [-1]
        if dof_component_map is None:
            dof_component_map = {}
            
        # From dolfin/function/LagrangeInterpolator.cpp,
        # method LagrangeInterpolator::extract_dof_component_map
        # Copyright (C) 2014 Mikael Mortensen
        if V.num_sub_spaces() == 0:
            # Extract sub dofmaps recursively and store dof to component map
            collapsed_dofs = V.dofmap().collapse(V.mesh())[1].values()
            component[0] += 1
            for collapsed_dof in collapsed_dofs:
                if not recursive: # space with only one component, do not print it
                    dof_component_map[collapsed_dof] = -1
                else:
                    dof_component_map[collapsed_dof] = component[0]
        else:
            for i in range(V.num_sub_spaces()):
                Vs = V.sub(i)
                _get_local_dof_to_component_map(Vs, component, dof_component_map, True)
        
        if not recursive:
            _get_local_dof_to_component_map._storage[V] = dof_component_map
    if not recursive:
        return _get_local_dof_to_component_map._storage[V]
    else:
        return None
_get_local_dof_to_component_map._storage = dict()
