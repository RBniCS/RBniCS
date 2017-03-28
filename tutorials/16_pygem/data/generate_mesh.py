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
rectangle = Rectangle(Point(-1., -2.5), Point(3, 2.5))
NACA0012_n_points = 64
NACA0012_points = list()
for n in range(2*NACA0012_n_points):
    if n < NACA0012_n_points:
        x = n*1./NACA0012_n_points
        sign = + 1.
    else:
        x = 1. + (NACA0012_n_points - n)*1./NACA0012_n_points
        sign = - 1.
    y = 5*0.12*(0.2969*sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    theta = -5*pi/180
    rot_x = cos(theta)*x - sin(theta)*sign*y
    rot_y = sin(theta)*x + cos(theta)*sign*y
    NACA0012_points.append( Point(rot_x, rot_y) )
NACA0012_points = NACA0012_points[::-1] # counter clockwise order
NACA0012 = Polygon(NACA0012_points)
domain = rectangle - NACA0012
mesh = generate_mesh(domain, 46)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
plot(subdomains)
interactive()

# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] + 1.) < DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 3.) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] + 2.5) < DOLFIN_EPS
                
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 2.5) < DOLFIN_EPS
                
class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
naca_boundary = AllBoundary()
naca_boundary.mark(boundaries, 1) # this will mark all the boundary, but it will be overwritten later
bottom = Bottom()
bottom.mark(boundaries, 2)
left = Left()
left.mark(boundaries, 3)
top = Top()
top.mark(boundaries, 4)
right = Right()
right.mark(boundaries, 5)
plot(boundaries)
interactive()

# Save
File("naca0012.xml") << mesh
File("naca0012_physical_region.xml") << subdomains
File("naca0012_facet_region.xml") << boundaries
