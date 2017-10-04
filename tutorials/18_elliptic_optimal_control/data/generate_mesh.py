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

def generate_mesh1():
    pass # Uses the same mesh as tutorial 04
    
def generate_mesh2():
    # Create mesh
    rectangle = Rectangle(Point(0., 0.), Point(2.5, 1.))
    domain = rectangle
    subdomain = dict()
    subdomain[1] = Rectangle(Point(0.2, 0.3), Point(0.8, 0.7))
    subdomain[2] = Rectangle(Point(1.2, 0.3), Point(2.5, 0.7))
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
            return on_boundary and abs(x[0] - 2.5) < DOLFIN_EPS 
            
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
    left = Left()
    left.mark(boundaries, 1)
    bottom1 = Bottom(0., 1.) 
    bottom1.mark(boundaries, 1)
    top1 = Top(0., 1.) 
    top1.mark(boundaries, 1)
    bottom2 = Bottom(1., 2.5) 
    bottom2.mark(boundaries, 2)
    top2 = Top(1., 2.5)
    top2.mark(boundaries, 2)
    right = Right()
    right.mark(boundaries, 3)

    # Save
    File("mesh2.xml") << mesh
    File("mesh2_physical_region.xml") << subdomains
    File("mesh2_facet_region.xml") << boundaries
    XDMFFile("mesh2.xdmf").write(mesh)
    XDMFFile("mesh2_physical_region.xdmf").write(subdomains)
    XDMFFile("mesh2_facet_region.xdmf").write(boundaries)
    
generate_mesh1()
generate_mesh2()
