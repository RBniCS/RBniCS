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

from dolfin import Cell, cells, compile_extension_module, DEBUG, Facet, facets, Function, FunctionSpace, LagrangeInterpolator, Mesh, MeshEditor, MeshFunction, log, parameters, Vertex, vertices
from numpy import abs, array, isclose, float, logical_and, max, round, uintp, unique, where, zeros
from mpi4py.MPI import SUM

# Implement an extended version of cbcpost create_submesh that:
# a) as cbcpost version (and in contrast to standard dolfin) also works in parallel 
# b) works for number of cells less than number of processors, by arbitrarily adding a cell on empty processors
# c) also assigns shared_entities(0). This is essential because otherwise mid-dofs on shared facets will be duplicated!
# d) assigns a global index of vertices and cells which is independent from the number of processors
# e) stores map between mesh and submesh cells, facets and vertices
# Part of this code is taken from cbcpost/utils/submesh.py, with the following copyright information:
# Copyright (C) 2010-2014 Simula Research Laboratory
def create_submesh(mesh, markers, marker_id):
    mpi_comm = mesh.mpi_comm().tompi4py()
    ## 1. Extract marked cells ##
    # Dolfin does not support a distributed mesh that is empty on some processes.
    # cbcpost gets around this by moving a single cell from the a non-empty processor to an empty one.
    # Note that, however, this cannot work if the number of marked cell is less than the number of processors.
    # In the interest of considering this case, we enable at least one cell (arbitrarily) on each processor.
    # We find this solution acceptable for our purposes, despite the increase of the reduced mesh size,
    # since we are never actually interested in solving a PDE on the reduced mesh, but rather only in
    # assemblying tensors on it and extract their values at some locations.
    backup_first_marker_id = None
    if not marker_id in markers.array():
        backup_first_marker_id = markers.array()[0]
        markers.array()[0] = marker_id
    assert marker_id in markers.array()
    
    ## 2. Create submesh ##
    submesh = Mesh(mesh.mpi_comm())
    mesh_editor = MeshEditor()
    mesh_editor.open(submesh,
                     mesh.ufl_cell().cellname(),
                     mesh.ufl_cell().topological_dimension(),
                     mesh.ufl_cell().geometric_dimension())
    # Extract cells from mesh with specified marker_id
    mesh_cell_indices = where(markers.array() == marker_id)[0]
    mesh_cells = mesh.cells()[mesh_cell_indices]
    mesh_global_cell_indices = sorted([mesh.topology().global_indices(mesh.topology().dim())[cell_index] for cell_index in mesh_cell_indices])
    # Get vertices of extracted cells
    mesh_vertex_indices = unique(mesh_cells.flatten())
    mesh_global_vertex_indices = sorted([mesh.topology().global_indices(0)[vertex_index] for vertex_index in mesh_vertex_indices])
    # Number vertices in a way which is independent from the number of processors. To do so ...
    # ... first of all collect all vertices from all processors
    allgathered_mesh_global_vertex_indices__non_empty_processors = list()
    allgathered_mesh_global_vertex_indices__empty_processors = list()
    for r in range(mpi_comm.size):
        backup_first_marker_id_r = mpi_comm.bcast(backup_first_marker_id, root=r)
        if backup_first_marker_id_r is None:
            allgathered_mesh_global_vertex_indices__non_empty_processors.extend(mpi_comm.bcast(mesh_global_vertex_indices, root=r))
        else:
            allgathered_mesh_global_vertex_indices__empty_processors.extend(mpi_comm.bcast(mesh_global_vertex_indices, root=r))
    allgathered_mesh_global_vertex_indices__non_empty_processors = \
        sorted(unique(allgathered_mesh_global_vertex_indices__non_empty_processors))
    allgathered_mesh_global_vertex_indices__empty_processors = \
        sorted(unique(allgathered_mesh_global_vertex_indices__empty_processors))
    # ... then create a dict that will contain the map from mesh global vertex index to submesh global vertex index.
    # ... Here make sure to number first "real" vertices (those coming from non empty processors), since the other ones
    # ... are just a side effect of the current partitioning!
    allgathered_mesh_to_submesh_vertex_global_indices = dict()
    _submesh_vertex_global_index = 0
    for mesh_vertex_global_index in allgathered_mesh_global_vertex_indices__non_empty_processors:
        assert mesh_vertex_global_index not in allgathered_mesh_to_submesh_vertex_global_indices
        allgathered_mesh_to_submesh_vertex_global_indices[mesh_vertex_global_index] = _submesh_vertex_global_index
        _submesh_vertex_global_index += 1
    for mesh_vertex_global_index in allgathered_mesh_global_vertex_indices__empty_processors:
        if mesh_vertex_global_index not in allgathered_mesh_to_submesh_vertex_global_indices:
            allgathered_mesh_to_submesh_vertex_global_indices[mesh_vertex_global_index] = _submesh_vertex_global_index
            _submesh_vertex_global_index += 1
    # Number cells in a way which is independent from the number of processors. To do so ...
    # ... first of all collect all cells from all processors
    allgathered_mesh_global_cell_indices__non_empty_processors = list()
    allgathered_mesh_global_cell_indices__empty_processors = list()
    for r in range(mpi_comm.size):
        backup_first_marker_id_r = mpi_comm.bcast(backup_first_marker_id, root=r)
        if backup_first_marker_id_r is None:
            allgathered_mesh_global_cell_indices__non_empty_processors.extend(mpi_comm.bcast(mesh_global_cell_indices, root=r))
        else:
            allgathered_mesh_global_cell_indices__empty_processors.extend(mpi_comm.bcast(mesh_global_cell_indices, root=r))
    allgathered_mesh_global_cell_indices__non_empty_processors = \
        sorted(unique(allgathered_mesh_global_cell_indices__non_empty_processors))
    allgathered_mesh_global_cell_indices__empty_processors = \
        sorted(unique(allgathered_mesh_global_cell_indices__empty_processors))
    # ... then create a dict that will contain the map from mesh global cell index to submesh global cell index.
    # ... Here make sure to number first "real" vertices (those coming from non empty processors), since the other ones
    # ... are just a side effect of the current partitioning!
    allgathered_mesh_to_submesh_cell_global_indices = dict()
    _submesh_cell_global_index = 0
    for mesh_cell_global_index in allgathered_mesh_global_cell_indices__non_empty_processors:
        assert mesh_cell_global_index not in allgathered_mesh_to_submesh_cell_global_indices
        allgathered_mesh_to_submesh_cell_global_indices[mesh_cell_global_index] = _submesh_cell_global_index
        _submesh_cell_global_index += 1
    for mesh_cell_global_index in allgathered_mesh_global_cell_indices__empty_processors:
        assert mesh_cell_global_index not in allgathered_mesh_to_submesh_cell_global_indices
        allgathered_mesh_to_submesh_cell_global_indices[mesh_cell_global_index] = _submesh_cell_global_index
        _submesh_cell_global_index += 1
    # Also create a mapping from mesh local vertex index to submesh local vertex index.
    mesh_to_submesh_vertex_local_indices = dict(zip(mesh_vertex_indices, range(len(mesh_vertex_indices))))
    # Also create a mapping from mesh local cell index to submesh local cell index.
    mesh_to_submesh_cell_local_indices = dict(zip(mesh_cell_indices, range(len(mesh_cell_indices))))
    # Now, define submesh cells
    submesh_cells = list()
    for i, c in enumerate(mesh_cells):
        submesh_cells.append([mesh_to_submesh_vertex_local_indices[j] for j in c])
    # Store vertices as submesh_vertices[local_index] = (global_index, coordinates)
    submesh_vertices = dict()
    for mesh_vertex_local_index, submesh_vertex_local_index in mesh_to_submesh_vertex_local_indices.items():
        submesh_vertices[submesh_vertex_local_index] = (
            allgathered_mesh_to_submesh_vertex_global_indices[mesh.topology().global_indices(0)[mesh_vertex_local_index]],
            mesh.coordinates()[mesh_vertex_local_index]
        )
    # Collect the global number of vertices and cells
    global_num_cells = mpi_comm.allreduce(len(submesh_cells), op=SUM)
    global_num_vertices = len(allgathered_mesh_to_submesh_vertex_global_indices)
    # Fill in mesh_editor
    mesh_editor.init_vertices_global(len(submesh_vertices), global_num_vertices)
    mesh_editor.init_cells_global(len(submesh_cells), global_num_cells)
    for local_index, cell_vertices in enumerate(submesh_cells):
        mesh_editor.add_cell(local_index, *cell_vertices)
    for local_index, (global_index, coordinates) in submesh_vertices.items():
        mesh_editor.add_vertex_global(local_index, global_index, coordinates)
    mesh_editor.close()
    # Initialize topology
    submesh.topology().init(0, len(submesh_vertices), global_num_vertices)
    submesh.topology().init(mesh.ufl_cell().topological_dimension(), len(submesh_cells), global_num_cells)
    # Correct the global index of cells
    for local_index in range(len(submesh_cells)):
        submesh.topology().set_global_index(
            submesh.topology().dim(), 
            local_index, 
            allgathered_mesh_to_submesh_cell_global_indices[ mesh_global_cell_indices[local_index] ]
        )
    
    ## 3. Store (local) mesh to/from submesh map for cells, facets and vertices ##
    # Cells
    submesh.mesh_to_submesh_cell_local_indices = mesh_to_submesh_cell_local_indices
    submesh.submesh_to_mesh_cell_local_indices = mesh_cell_indices
    # Vertices
    submesh.mesh_to_submesh_vertex_local_indices = mesh_to_submesh_vertex_local_indices
    submesh.submesh_to_mesh_vertex_local_indices = mesh_vertex_indices
    # Facets
    mesh_vertices_to_mesh_facets = dict()
    mesh_facets_to_mesh_vertices = dict()
    for mesh_cell_index in mesh_cell_indices:
        mesh_cell = Cell(mesh, mesh_cell_index)
        for mesh_facet in facets(mesh_cell):
            mesh_facet_vertices = list()
            for mesh_facet_vertex in vertices(mesh_facet):
                mesh_facet_vertices.append(mesh_facet_vertex.index())
            mesh_facet_vertices = tuple(sorted(mesh_facet_vertices))
            if mesh_facet_vertices in mesh_vertices_to_mesh_facets:
                assert mesh_vertices_to_mesh_facets[mesh_facet_vertices] == mesh_facet.index()
            else:
                mesh_vertices_to_mesh_facets[mesh_facet_vertices] = mesh_facet.index()
            if mesh_facet.index() in mesh_facets_to_mesh_vertices:
                assert mesh_facets_to_mesh_vertices[mesh_facet.index()] == mesh_facet_vertices
            else:
                mesh_facets_to_mesh_vertices[mesh_facet.index()] = mesh_facet_vertices
    submesh_vertices_to_submesh_facets = dict()
    submesh_facets_to_submesh_vertices = dict()
    for submesh_facet in facets(submesh):
        submesh_facet_vertices = list()
        for submesh_facet_vertex in vertices(submesh_facet):
            submesh_facet_vertices.append(submesh_facet_vertex.index())
        submesh_facet_vertices = tuple(sorted(submesh_facet_vertices))
        assert submesh_facet_vertices not in submesh_vertices_to_submesh_facets
        submesh_vertices_to_submesh_facets[submesh_facet_vertices] = submesh_facet.index()
        assert submesh_facet.index() not in submesh_facets_to_submesh_vertices
        submesh_facets_to_submesh_vertices[submesh_facet.index()] = submesh_facet_vertices
    mesh_to_submesh_facets_local_indices = dict()
    for (mesh_facet_index, mesh_vertices) in mesh_facets_to_mesh_vertices.iteritems():
        submesh_vertices = tuple(sorted([submesh.mesh_to_submesh_vertex_local_indices[mesh_vertex] for mesh_vertex in mesh_vertices]))
        submesh_facet_index = submesh_vertices_to_submesh_facets[submesh_vertices]
        mesh_to_submesh_facets_local_indices[mesh_facet_index] = submesh_facet_index
    submesh_to_mesh_facets_local_indices = dict()
    for (submesh_facet_index, submesh_vertices) in submesh_facets_to_submesh_vertices.iteritems():
        mesh_vertices = tuple(sorted([submesh.submesh_to_mesh_vertex_local_indices[submesh_vertex] for submesh_vertex in submesh_vertices]))
        mesh_facet_index = mesh_vertices_to_mesh_facets[mesh_vertices]
        submesh_to_mesh_facets_local_indices[submesh_facet_index] = mesh_facet_index
    submesh.mesh_to_submesh_facet_local_indices = mesh_to_submesh_facets_local_indices
    submesh.submesh_to_mesh_facet_local_indices = list()
    assert min(submesh_to_mesh_facets_local_indices.keys()) == 0
    assert max(submesh_to_mesh_facets_local_indices.keys()) == len(submesh_to_mesh_facets_local_indices.keys()) - 1
    for submesh_facet_index in range(len(submesh_to_mesh_facets_local_indices)):
        submesh.submesh_to_mesh_facet_local_indices.append(submesh_to_mesh_facets_local_indices[submesh_facet_index])
        
    ## 4. Assign shared vertices ##
    shared_entities_dimensions = {
        "vertex": 0,
        "facet":  submesh.topology().dim() - 1,
        "cell":   submesh.topology().dim()
    }
    shared_entities_submesh_to_mesh_map = {
        "vertex": submesh.submesh_to_mesh_vertex_local_indices,
        "facet" : submesh.submesh_to_mesh_facet_local_indices,
        "cell"  : submesh.submesh_to_mesh_cell_local_indices
    }
    shared_entities_class = {
        "vertex": Vertex,
        "facet" : Facet,
        "cell"  : Cell
    }
    shared_entities_iterator = {
        "vertex": vertices,
        "facet" : facets,
        "cell"  : cells
    }
    for entity in ["vertex", "facet", "cell"]: # do not use .keys() because the order is important
        if entity == "facet":
            # Make sure to initialize global indices for facets. I have yet to find a reliable way to do that,
            # because mesh.init() seems to compute connectivies but not to initialize global indices.
            # In the meantime, obtain this as a side effect of a function space declaration
            _ = FunctionSpace(mesh, "CG", 2)
            _ = FunctionSpace(submesh, "CG", 2)
        dim = shared_entities_dimensions[entity]
        submesh_to_mesh_map = shared_entities_submesh_to_mesh_map[entity]
        class_ = shared_entities_class[entity]
        iterator = shared_entities_iterator[entity]
        # Get shared entities from mesh. A subset of these will end being shared entities also the submesh
        # (thanks to the fact that we do not redistribute cells from one processor to another)
        if mpi_comm.size > 1: # some entities may not be initialized in serial, since they are not needed
            assert mesh.topology().have_shared_entities(dim), "Mesh shared entities have not been initialized for dimension " + str(dim)
        if mesh.topology().have_shared_entities(dim): # always true in parallel (when really needed)
            mesh_shared_entities = mesh.topology().shared_entities(dim)
            # Discard those entity which were not part of the marker selection
            mesh_shared_entity_local_indices = set(submesh_to_mesh_map).intersection(set(mesh_shared_entities.keys()))
            # However, it may happen that an entity which has been selected is not shared anymore because only one of
            # the sharing processes has it in the submesh. For instance, consider the case
            # of two cells across the interface (located on a facet f) between two processors. It may happen that
            # only one of the two cells is selected: the facet f and its vertices are not shared anymore!
            # For this reason, we create a new dict from global entity index to processors sharing them
            # (which will be a subset of mesh_shared_entities, which retains original sharing processors). Thus ...
            # ... first of all get global indices corresponding to local entities
            assert submesh.topology().have_global_indices(dim), "Submesh global indices have not been initialized for dimension " + str(dim)
            submesh_local_entities_global_index = list()
            submesh_local_entities_global_to_local_index = dict()
            for entity in iterator(submesh):
                submesh_local_entities_global_index.append(entity.global_index())
                submesh_local_entities_global_to_local_index[entity.global_index()] = entity.index()
            # ... then gather all global indices from all processors
            gathered__submesh_local_entities_global_index = list() # over processor id
            for r in range(mpi_comm.size):
                gathered__submesh_local_entities_global_index.append( mpi_comm.bcast(submesh_local_entities_global_index, root=r) )
            # ... then create dict from global index to processors sharing it
            submesh_shared_entities__global = dict()
            for r in range(mpi_comm.size):
                for global_entity_index in gathered__submesh_local_entities_global_index[r]:
                    if global_entity_index not in submesh_shared_entities__global:
                        submesh_shared_entities__global[global_entity_index] = list()
                    submesh_shared_entities__global[global_entity_index].append(r)
            # ... and finally popuplate shared entities dict, which is the same as the dict above except that
            # the current processor rank is removed and a local indexing is used
            submesh_shared_entities = dict() # from local index to list of integers
            for (global_entity_index, processors) in submesh_shared_entities__global.iteritems():
                if (
                    mpi_comm.rank in processors  # only local entities
                        and 
                    len(processors) > 1 # it was still shared after submesh extraction
                ):
                    other_processors_list = list(processors)
                    other_processors_list.remove(mpi_comm.rank)
                    other_processors = array(other_processors_list, dtype=uintp)
                    submesh_shared_entities[ submesh_local_entities_global_to_local_index[global_entity_index] ] = other_processors

            # Need an extension module to populate shared_entities because in python each call to shared_entities
            # returns a temporary.
            cpp_code = """
                void set_shared_entities(Mesh & submesh, std::size_t idx, const Array<std::size_t>& other_processes, std::size_t dim)
                {
                    std::set<unsigned int> set_other_processes;
                    for (std::size_t i(0); i < other_processes.size(); i++)
                        set_other_processes.insert(other_processes[i]);
                    submesh.topology().shared_entities(dim)[idx] = set_other_processes;
                }
            """
            set_shared_entities = compile_extension_module(cpp_code).set_shared_entities
            for (submesh_entity_local_index, other_processors) in submesh_shared_entities.iteritems():
                set_shared_entities(submesh, submesh_entity_local_index, other_processors, dim)
                
            log(DEBUG, "Local indices of shared entities for dimension " + str(dim) + ": " + str(submesh.topology().shared_entities(0).keys()))
            log(DEBUG, "Global indices of shared entities for dimension " + str(dim) + ": " + str([class_(submesh, local_index).global_index() for local_index in submesh.topology().shared_entities(dim).keys()]))
    
    ## 5. Restore backup_first_marker_id and return ##
    if backup_first_marker_id is not None:
        markers.array()[0] = backup_first_marker_id
    return submesh
       
def create_submesh_subdomains(mesh, submesh, mesh_subdomains):
    assert mesh_subdomains is None or (isinstance(mesh_subdomains, list) and len(mesh_subdomains) > 0)
    if mesh_subdomains is None:
        return None
    submesh_subdomains = list()
    # Create submesh subdomains
    for mesh_subdomain in mesh_subdomains:
        submesh_subdomain = MeshFunction("size_t", submesh, mesh_subdomain.dim())
        submesh_subdomain.set_all(0)
        assert submesh_subdomain.dim() in (submesh.topology().dim(), submesh.topology().dim() - 1)
        if submesh_subdomain.dim() == submesh.topology().dim():
            for submesh_cell in cells(submesh):
                submesh_subdomain.array()[submesh_cell.index()] = mesh_subdomain.array()[submesh.submesh_to_mesh_cell_local_indices[submesh_cell.index()]]
        elif submesh_subdomain.dim() == submesh.topology().dim() - 1:
            for submesh_facet in facets(submesh):
                submesh_subdomain.array()[submesh_facet.index()] = mesh_subdomain.array()[submesh.submesh_to_mesh_facet_local_indices[submesh_facet.index()]]
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in create_submesh_subdomains.")
        submesh_subdomains.append(submesh_subdomain)
    return submesh_subdomains

def mesh_dofs_to_submesh_dofs(mesh_V, submesh_V):
    inverse_map = submesh_dofs_to_mesh_dofs(submesh_V, mesh_V)
    return dict(zip(inverse_map.values(), inverse_map.keys()))
    
def submesh_dofs_to_mesh_dofs(submesh_V, mesh_V):
    if mesh_V.ufl_element().family() == "Mixed":
        assert mesh_V.num_sub_spaces() == submesh_V.num_sub_spaces()
        for i in range(mesh_V.num_sub_spaces()):
            assert mesh_V.sub(i).ufl_element().family() == "Lagrange", "The current implementation of mapping between dofs relies on LagrangeInterpolator"
            assert submesh_V.sub(i).ufl_element().family() == "Lagrange", "The current implementation of mapping between dofs relies on LagrangeInterpolator"
    else:
        assert mesh_V.ufl_element().family() == "Lagrange", "The current implementation of mapping between dofs relies on LagrangeInterpolator"
        assert submesh_V.ufl_element().family() == "Lagrange", "The current implementation of mapping between dofs relies on LagrangeInterpolator"
    
    previous_allow_extrapolation_value = parameters["allow_extrapolation"]
    parameters["allow_extrapolation"] = True
    
    interpolator = LagrangeInterpolator()
    mesh_V_ownership_range = array(range(*mesh_V.dofmap().ownership_range()), dtype=float)
    mesh_V_dofs = Function(mesh_V)
    mesh_V_dofs.vector().set_local(mesh_V_ownership_range)
    mesh_V_dofs.vector().apply("")
    mesh_V_dofs_restricted_to_submesh_V = Function(submesh_V)
    interpolator.interpolate(mesh_V_dofs_restricted_to_submesh_V, mesh_V_dofs)
    mesh_V_dofs_restricted_to_submesh_V = mesh_V_dofs_restricted_to_submesh_V.vector().array()
    submesh_V_local_indices = where(logical_and(
        mesh_V_dofs_restricted_to_submesh_V >= - 0.5,
        logical_and(
            mesh_V_dofs_restricted_to_submesh_V <= mesh_V.dim() + 0.5, 
            abs(mesh_V_dofs_restricted_to_submesh_V - round(mesh_V_dofs_restricted_to_submesh_V)) < 1.e-10
        )
    ))[0]
    submesh_V_global_indices = [submesh_V.dofmap().local_to_global_index(local_dof) for local_dof in submesh_V_local_indices]
    mesh_V_global_indices = round(mesh_V_dofs_restricted_to_submesh_V[submesh_V_local_indices]).astype('i')
    assert len(unique(mesh_V_global_indices)) == len(mesh_V_global_indices)
    assert len(mesh_V_global_indices) == submesh_V.dofmap().ownership_range()[1] - submesh_V.dofmap().ownership_range()[0]
    
    parameters["allow_extrapolation"] = previous_allow_extrapolation_value
    
    submesh_V_to_mesh_V_global_indices = dict(zip(submesh_V_global_indices, mesh_V_global_indices))
    
    mpi_comm = submesh_V.mesh().mpi_comm().tompi4py()
    allgathered_submesh_V_to_mesh_V_global_indices = mpi_comm.bcast(submesh_V_to_mesh_V_global_indices, root=0)
    for r in range(1, mpi_comm.size):
        allgathered_submesh_V_to_mesh_V_global_indices.update( mpi_comm.bcast(submesh_V_to_mesh_V_global_indices, root=r) )
    
    return allgathered_submesh_V_to_mesh_V_global_indices
    
