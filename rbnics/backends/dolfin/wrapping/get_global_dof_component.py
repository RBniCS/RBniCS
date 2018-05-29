# Copyright (C) 2015-2018 by the RBniCS authors
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
from dolfin import has_pybind11
if has_pybind11():
    from dolfin import compile_cpp_code
from rbnics.backends.dolfin.wrapping.get_global_dof_to_local_dof_map import get_global_dof_to_local_dof_map
from rbnics.utils.cache import cache

def get_global_dof_component(global_dof, V, global_to_local=None, local_dof_to_component=None):
    if global_to_local is None:
        global_to_local = get_global_dof_to_local_dof_map(V, V.dofmap())
    if local_dof_to_component is None:
        local_dof_to_component = _get_local_dof_to_component_map(V)
    
    mpi_comm = V.mesh().mpi_comm()
    if not has_pybind11():
        mpi_comm = mpi_comm.tompi4py()
    dof_component = None
    dof_component_processor = -1
    if global_dof in global_to_local:
        dof_component = local_dof_to_component[global_to_local[global_dof]]
        dof_component_processor = mpi_comm.rank
    dof_component_processor = mpi_comm.allreduce(dof_component_processor, op=MAX)
    assert dof_component_processor >= 0
    return mpi_comm.bcast(dof_component, root=dof_component_processor)
    
@cache
def _get_local_dof_to_component_map(V, component=None, dof_component_map=None, recursive=False):
    if component is None:
        component = [-1]
    if dof_component_map is None:
        dof_component_map = dict()
        
    # From dolfin/function/LagrangeInterpolator.cpp,
    # method LagrangeInterpolator::extract_dof_component_map
    # Copyright (C) 2014 Mikael Mortensen
    if V.num_sub_spaces() == 0:
        # Extract sub dofmaps recursively and store dof to component map
        if has_pybind11():
            cpp_code = """
                #include <pybind11/pybind11.h>
                #include <pybind11/stl.h>
                #include <dolfin/fem/DofMap.h>
                #include <dolfin/mesh/Mesh.h>
                
                std::vector<std::size_t> collapse_dofmap(std::shared_ptr<dolfin::DofMap> dofmap, std::shared_ptr<dolfin::Mesh> mesh)
                {
                    std::unordered_map<std::size_t, std::size_t> collapsed_map;
                    dofmap->collapse(collapsed_map, *mesh);
                    std::vector<std::size_t> collapsed_dofs;
                    collapsed_dofs.reserve(collapsed_map.size());
                    for (auto const& collapsed_map_item: collapsed_map)
                        collapsed_dofs.push_back(collapsed_map_item.second);
                    return collapsed_dofs;
                }
                
                PYBIND11_MODULE(SIGNATURE, m)
                {
                    m.def("collapse_dofmap", &collapse_dofmap);
                }
            """
            collapse_dofmap = compile_cpp_code(cpp_code).collapse_dofmap
            collapsed_dofs = collapse_dofmap(V.dofmap(), V.mesh())
        else:
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
        return dof_component_map
    else:
        return None
