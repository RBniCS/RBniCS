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
from mshr import *

# Create mesh
rectangle = Rectangle(Point(0., 0.), Point(2., 1.))
subdomain = dict()
subdomain[1] = Rectangle(Point(0., 0.), Point(1., 1.))
subdomain[2] = Rectangle(Point(1., 0.), Point(2., 1.))
domain = rectangle
for i, s in subdomain.items():
    domain.set_subdomain(i, subdomain[i])
mesh = generate_mesh(domain, 64)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Create boundaries
class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS

class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 2.) < DOLFIN_EPS

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
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
bottomLeft = Bottom(0., 1.)
bottomLeft.mark(boundaries, 1)
bottomRightLeft = Bottom(1., 1.5)
bottomRightLeft.mark(boundaries, 2)
bottomRightRight = Bottom(1.5, 2.)
bottomRightRight.mark(boundaries, 3)
right = Right()
right.mark(boundaries, 4)
topRightRight = Top(1.5, 2.)
topRightRight.mark(boundaries, 5)
topRightLeft = Top(1., 1.5)
topRightLeft.mark(boundaries, 6)
topLeft = Top(0., 1.)
topLeft.mark(boundaries, 7)
left = Left()
left.mark(boundaries, 8)

# Save
File("graetz.xml") << mesh
File("graetz_physical_region.xml") << subdomains
File("graetz_facet_region.xml") << boundaries
XDMFFile("graetz.xdmf").write(mesh)
XDMFFile("graetz_physical_region.xdmf").write(subdomains)
XDMFFile("graetz_facet_region.xdmf").write(boundaries)
