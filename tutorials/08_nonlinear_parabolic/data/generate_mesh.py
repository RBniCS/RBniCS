# Copyright (C) 2015-2019 by the RBniCS authors
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

# Create mesh
mesh = UnitIntervalMesh(1024)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)

# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left = Left()
left.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 2)

# Save
File("interval.xml") << mesh
File("interval_physical_region.xml") << subdomains
File("interval_facet_region.xml") << boundaries
XDMFFile("interval.xdmf").write(mesh)
XDMFFile("interval_physical_region.xdmf").write(subdomains)
XDMFFile("interval_facet_region.xdmf").write(boundaries)
