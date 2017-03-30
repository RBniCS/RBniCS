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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from mpi4py.MPI import MAX
from dolfin import assemble, inner, dP, TestFunction

def ufl_lagrange_interpolation(output, ufl_expression):
    V = output.function_space()
    if not V in ufl_lagrange_interpolation._test_function:
        ufl_lagrange_interpolation._test_function[V] = TestFunction(V)
    v = ufl_lagrange_interpolation._test_function[V]
    assemble(inner(v, ufl_expression)*dP, output.vector())
ufl_lagrange_interpolation._test_function = dict()
    
def get_global_dof_coordinates(global_dof, V):
    if not V in _prepare_global_dof_to_local_dof_map._storage:
        _prepare_global_dof_to_local_dof_map._storage[V] = _prepare_global_dof_to_local_dof_map(V)
    global_to_local = _prepare_global_dof_to_local_dof_map._storage[V]
    
    if not V in _prepare_local_dof_to_coordinates_map._storage:
        _prepare_local_dof_to_coordinates_map._storage[V] = _prepare_local_dof_to_coordinates_map(V)
    local_dof_to_coordinates = _prepare_local_dof_to_coordinates_map._storage[V]
    
    mpi_comm = V.mesh().mpi_comm().tompi4py()
    dof_coordinates = None
    dof_coordinates_processor = -1
    if global_dof in global_to_local:
        dof_coordinates = local_dof_to_coordinates[global_to_local[global_dof]]
        dof_coordinates_processor = mpi_comm.rank
    dof_coordinates_processor = mpi_comm.allreduce(dof_coordinates_processor, op=MAX)
    assert dof_coordinates_processor >= 0
    return mpi_comm.bcast(dof_coordinates, root=dof_coordinates_processor)
    
def get_global_dof_component(global_dof, V):
    if V.element().num_sub_elements() == 0:
        return -1
        
    if not V in _prepare_global_dof_to_local_dof_map._storage:
        _prepare_global_dof_to_local_dof_map._storage[V] = _prepare_global_dof_to_local_dof_map(V)
    global_to_local = _prepare_global_dof_to_local_dof_map._storage[V]
    
    if not V in _prepare_local_dof_to_component_map._storage:
        _prepare_local_dof_to_component_map._component = -1
        _prepare_local_dof_to_component_map._dof_component_map = dict()
        _prepare_local_dof_to_component_map(V)
        _prepare_local_dof_to_component_map._storage[V] = _prepare_local_dof_to_component_map._dof_component_map
    local_dof_to_component = _prepare_local_dof_to_component_map._storage[V]
    
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

def _prepare_global_dof_to_local_dof_map(V):
    local_to_global = V.dofmap().tabulate_local_to_global_dofs()
    local_size = V.dofmap().ownership_range()[1] - V.dofmap().ownership_range()[0]
    global_to_local = {global_:local for (local, global_) in enumerate(local_to_global) if local < local_size}
    return global_to_local
_prepare_global_dof_to_local_dof_map._storage = dict()
    
def _prepare_local_dof_to_coordinates_map(V):
    return V.tabulate_dof_coordinates().reshape((-1, V.mesh().ufl_cell().topological_dimension()))
_prepare_local_dof_to_coordinates_map._storage = dict()

def _prepare_local_dof_to_component_map(V):
    # From dolfin/function/LagrangeInterpolator.cpp,
    # method LagrangeInterpolator::extract_dof_component_map
    # Copyright (C) 2014 Mikael Mortensen
    if V.element().num_sub_elements() == 0:
        # Extract sub dofmaps recursively and store dof to component map
        collapsed_dofs = V.dofmap().collapse(V.mesh())[1].values()
        _prepare_local_dof_to_component_map._component += 1
        for collapsed_dof in collapsed_dofs:
            _prepare_local_dof_to_component_map._dof_component_map[collapsed_dof] = _prepare_local_dof_to_component_map._component
    else:
        for i in range(V.element().num_sub_elements()):
            Vs = V.extract_sub_space([i])
            _prepare_local_dof_to_component_map(Vs)
_prepare_local_dof_to_component_map._storage = dict()
    
