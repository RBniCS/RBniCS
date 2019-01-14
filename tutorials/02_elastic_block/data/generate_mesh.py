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
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
for i in range(3):
    for j in range(3):
        domain.set_subdomain(i + j*3 + 1, Rectangle(Point(i/3., j/3.), Point((i+1)/3., (j+1)/3.)))
mesh = generate_mesh(domain, 32)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Create boundaries
class Left(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
        
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Right(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
        
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Bottom(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
                
class Top(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
        
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
for i in range(3):
    left = Left(i/3., (i+1)/3.)
    left.mark(boundaries, 6)
    top = Top(i/3., (i+1)/3.)
    top.mark(boundaries, 5)
    right = Right(i/3., (i+1)/3.)
    right.mark(boundaries, i+2)
    bottom = Bottom(i/3., (i+1)/3.)
    bottom.mark(boundaries, 1)

# Save
File("elastic_block.xml") << mesh
File("elastic_block_physical_region.xml") << subdomains
File("elastic_block_facet_region.xml") << boundaries
XDMFFile("elastic_block.xdmf").write(mesh)
XDMFFile("elastic_block_physical_region.xdmf").write(subdomains)
XDMFFile("elastic_block_facet_region.xdmf").write(boundaries)
