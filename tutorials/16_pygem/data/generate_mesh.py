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
from naca import naca # available from https://github.com/dgorissen/naca
import numpy as np

# Create mesh
rectangle = Rectangle(Point(-1., -2.5), Point(3, 2.5))
naca_type = "0012"
naca_n_points = 200
naca_angle = - 5.0
naca_x, naca_y = naca(naca_type, naca_n_points)
naca_x = np.array(naca_x)
naca_y = np.array(naca_y)
naca_x_center = np.average(naca_x)
naca_y_center = np.average(naca_y)
naca_x -= naca_x_center
naca_x -= naca_y_center
naca_x_rot = naca_x*np.cos(naca_angle*pi/180) - naca_y*np.sin(naca_angle*pi/180)
naca_y_rot = naca_x*np.sin(naca_angle*pi/180) + naca_y*np.cos(naca_angle*pi/180)
naca_x_rot += naca_x_center + (naca_x.max() - naca_x.min())/2
naca_x_rot += naca_y_center + (naca_y.max() - naca_y.min())/2
naca_points = [Point(x, y) for (x, y) in zip(naca_x_rot, naca_y_rot)]
naca = Polygon(naca_points)
domain = rectangle - naca
mesh = generate_mesh(domain, 46)

# Refine the mesh around the airfoil
refinement_box = [(0., 2.), (-1., 1.)]
for refinements in range(1):
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    for cell in cells(mesh):
        p = cell.midpoint()
        if (
            (refinement_box[0][0] < p[0] < refinement_box[0][1]) 
                and
            (refinement_box[1][0] < p[1] < refinement_box[1][1])
        ):
            cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

# Plot mesh
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
