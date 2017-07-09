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

from dolfin import *
set_log_level(PROGRESS)
from fenicstools import DofMapPlotter
from rbnics.backends.dolfin.wrapping import convert_functionspace_to_submesh, convert_meshfunctions_to_submesh, create_submesh

mesh = UnitSquareMesh(3, 3)
assert MPI.size(mesh.mpi_comm()) in (1, 2, 3, 4)
# 1 processor        -> test serial case
# 2 and 3 processors -> test case where submesh in contained only on one processor
# 4 processors       -> test case where submesh is shared by two processors, resulting in shared vertices

subdomains = CellFunction("size_t", mesh, 0)
for c in cells(mesh):
    subdomains.array()[c.index()] = c.global_index()
#plot(subdomains, interactive=True)

boundaries = FacetFunction("size_t", mesh, 0)
for f in facets(mesh):
    boundaries.array()[f.index()] = 0
    for v in vertices(f):
        boundaries.array()[f.index()] += v.global_index()
#plot(boundaries, interactive=True)

markers = CellFunction("bool", mesh, False)
hdf = HDF5File(mesh.mpi_comm(), "data/test_create_submesh_markers.h5", "r")
hdf.read(markers, "/cells")
#plot(markers, interactive=True)

submesh = create_submesh(mesh, markers)
#plot(submesh, interactive=True)

[submesh_subdomains, submesh_boundaries] = convert_meshfunctions_to_submesh(mesh, submesh, [subdomains, boundaries])
#plot(submesh_subdomains, interactive=True)
#plot(submesh_boundaries, interactive=True)

# A dof map plotter will be opened in a few lines. You can use it to do the following checks, denoted by a), b), etc.

# a) compare cell numbers in mesh and reduced mesh by pressing C in dof map plotter. Double check the following maps:
log(PROGRESS, "Mesh to submesh cell global indices")
for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_cell_local_indices.iteritems():
    mesh_global_index = mesh.topology().global_indices(mesh.topology().dim())[mesh_local_index]
    submesh_global_index = submesh.topology().global_indices(submesh.topology().dim())[submesh_local_index]
    log(PROGRESS, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
log(PROGRESS, "Submesh to mesh cell global indices")
for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_cell_local_indices):
    submesh_global_index = submesh.topology().global_indices(submesh.topology().dim())[submesh_local_index]
    mesh_global_index = mesh.topology().global_indices(mesh.topology().dim())[mesh_local_index]
    log(PROGRESS, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))

# b) compare facet numbers in mesh and reduced mesh by pressing T in dof map plotter. Double check the following maps:
log(PROGRESS, "Mesh to submesh facet global indices")
for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_facet_local_indices.iteritems():
    mesh_global_index = mesh.topology().global_indices(mesh.topology().dim() - 1)[mesh_local_index]
    submesh_global_index = submesh.topology().global_indices(submesh.topology().dim() - 1)[submesh_local_index]
    log(PROGRESS, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
log(PROGRESS, "Submesh to mesh facet global indices")
for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_facet_local_indices):
    submesh_global_index = submesh.topology().global_indices(submesh.topology().dim() - 1)[submesh_local_index]
    mesh_global_index = mesh.topology().global_indices(mesh.topology().dim() - 1)[mesh_local_index]
    log(PROGRESS, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))

# c) compare vertex numbers in mesh and reduced mesh by pressing V in dof map plotter. Double check the following maps:
log(PROGRESS, "Mesh to submesh vertex global indices")
for (mesh_local_index, submesh_local_index) in submesh.mesh_to_submesh_vertex_local_indices.iteritems():
    mesh_global_index = mesh.topology().global_indices(0)[mesh_local_index]
    submesh_global_index = submesh.topology().global_indices(0)[submesh_local_index]
    log(PROGRESS, "\t" + str(mesh_global_index) + " -> " + str(submesh_global_index))
log(PROGRESS, "Submesh to mesh vertex global indices")
for (submesh_local_index, mesh_local_index) in enumerate(submesh.submesh_to_mesh_vertex_local_indices):
    submesh_global_index = submesh.topology().global_indices(0)[submesh_local_index]
    mesh_global_index = mesh.topology().global_indices(0)[mesh_local_index]
    log(PROGRESS, "\t" + str(submesh_global_index) + " -> " + str(mesh_global_index))
    
# d) print shared indices
dim_to_text = {
    submesh.topology().dim(): "cells",
    submesh.topology().dim() - 1: "facets",
    0: "vertices"
}
for dim in [submesh.topology().dim(), submesh.topology().dim() - 1, 0]:
    log(PROGRESS, "Submesh shared indices for " + str(dim_to_text[dim]))
    log(PROGRESS, str(submesh.topology().shared_entities(dim)))

# ~~~ Elliptic case ~~~ #
log(PROGRESS, "~~~ Elliptic case ~~~")
V = FunctionSpace(mesh, "CG", 2)
(submesh_V, mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs) = convert_functionspace_to_submesh(V, submesh, markers)

# e) compare dof numbers in mesh and reduced mesh by pressing D in dof map plotter. Double check the following maps:
log(PROGRESS, "Mesh to submesh dofs")
log(PROGRESS, "Local mesh dofs ownership range: " + str(V.dofmap().ownership_range()))
for (mesh_dof, submesh_dof) in mesh_dofs_to_submesh_dofs.iteritems():
    log(PROGRESS, "\t" + str(mesh_dof) + " -> " + str(submesh_dof))
log(PROGRESS, "Submesh to mesh dofs")
log(PROGRESS, "Local submesh dofs ownership range: " + str(submesh_V.dofmap().ownership_range()))
for (submesh_dof, mesh_dof) in submesh_dofs_to_mesh_dofs.iteritems():
    log(PROGRESS, "\t" + str(submesh_dof) + " -> " + str(mesh_dof))
    
# In reduced function space dof map plotter:
# any processors -> f) press C to double check that the cell numbering is independent on the number of processors.
#                   g) moreover, processors with fake cells should have the largest numbering.
# 4 processors   -> h) press D to double check that the dofs numbering on the interface among processors is the same.
#                      In order to see the global dof id you will need to change line 155 of
#                      fenicstools/dofmapplotter/dofhandler.py with
#                           dof = self.dofmaps[j].local_to_global_index(dof)
#                   i) press T to double check that the facet numbering on the interface among processors is the same
# 

# Open dof map plotters
"""
dmp = DofMapPlotter(V)
dmp.plot()
dmp.show()

dmp = DofMapPlotter(submesh_V)
dmp.plot()
dmp.show()
"""

# ~~~ Mixed case ~~~ #
log(PROGRESS, "~~~ Mixed case ~~~")
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)
(submesh_V, mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs) = convert_functionspace_to_submesh(V, submesh, markers)

# e) compare dof numbers in mesh and reduced mesh by pressing D in dof map plotter. Double check the following maps:
log(PROGRESS, "Mesh to submesh dofs")
log(PROGRESS, "Local mesh dofs ownership range: " + str(V.dofmap().ownership_range()))
for (mesh_dof, submesh_dof) in mesh_dofs_to_submesh_dofs.iteritems():
    log(PROGRESS, "\t" + str(mesh_dof) + " -> " + str(submesh_dof))
log(PROGRESS, "Submesh to mesh dofs")
log(PROGRESS, "Local submesh dofs ownership range: " + str(submesh_V.dofmap().ownership_range()))
for (submesh_dof, mesh_dof) in submesh_dofs_to_mesh_dofs.iteritems():
    log(PROGRESS, "\t" + str(submesh_dof) + " -> " + str(mesh_dof))
    
# also check h) as above

# Open dof map plotters
"""
dmp = DofMapPlotter(V)
dmp.plot()
dmp.show()

dmp = DofMapPlotter(submesh_V)
dmp.plot()
dmp.show()
"""
