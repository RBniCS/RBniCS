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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

import os
from dolfin import *
from rbnics.backends.fenics.wrapping import create_submesh

# Make output directory, if necessary
try: 
    os.makedirs("test_create_submesh_shared_entities.output_dir")
except OSError:
    if not os.path.isdir("test_create_submesh_shared_entities.output_dir"):
        raise

mesh = Mesh()
input_file = HDF5File(mesh.mpi_comm(), "data/test_create_submesh_shared_entities_mesh.h5", "r")
input_file.read(mesh, "/mesh", False)
cells = CellFunction("size_t", mesh)
input_file.read(cells, "/cells")

assert MPI.size(mesh.mpi_comm()) in (1, 2, 3, 4, 5)
# 1 processor  -> test serial case
# 2 processors -> test a parallel case with no submesh shared entities and no submesh elements close to processors interface
# 3 processors -> test a parallel case with no submesh shared entities but a facet and a vertex are close to processors interface
# 4 processors -> test a parallel case with no submesh shared entities but a facet is close to processors interface
# 5 processors -> test a parallel case with submesh shared entities (a FE patch split accross two processors)

#plot(mesh, interactive=True)
#plot(cells, interactive=True)

submesh = create_submesh(mesh, cells, 1)
#plot(submesh, interactive=True)

output_subfile = HDF5File(submesh.mpi_comm(), "test_create_submesh_shared_entities.output_dir/submesh.h5", "w")
output_subfile.write(submesh, "/submesh")
output_subfile.close()

# Read back in the submesh. In the past bugs where highlighted when calling hdf5 writer with incorrect shared entities,
# resulting in vertices in a clearly wrong location when reading back in the mesh
submesh2 = Mesh()
input_subfile = HDF5File(submesh2.mpi_comm(), "test_create_submesh_shared_entities.output_dir/submesh.h5", "r")
input_subfile.read(submesh2, "/submesh", False)
#plot(submesh2, interactive=True)

