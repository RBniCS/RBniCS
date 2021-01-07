# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import compile_cpp_code
from rbnics.utils.cache import cache


@cache
def get_local_dof_to_component_map(V):
    return _get_local_dof_to_component_map(V)


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
        cpp_code = """
            #include <pybind11/pybind11.h>
            #include <pybind11/stl.h>
            #include <dolfin/fem/DofMap.h>
            #include <dolfin/mesh/Mesh.h>

            std::vector<std::size_t> collapse_dofmap(std::shared_ptr<dolfin::DofMap> dofmap,
                                                     std::shared_ptr<dolfin::Mesh> mesh)
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
        component[0] += 1
        for collapsed_dof in collapsed_dofs:
            if not recursive:  # space with only one component, do not print it
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
