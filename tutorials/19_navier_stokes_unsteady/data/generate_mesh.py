# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Geometrical parameters
# ... of outer rectangle
L = 2.5
H = 0.41
# ... of circle
C_x = 0.4
C_y = 0.205
r = 0.05

# Create mesh
rectangle = Rectangle(Point(0., 0.), Point(L, H))
circle = Circle(Point(C_x, C_y), r, segments=32)
domain = rectangle - circle
mesh = generate_mesh(domain, 100)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)


# Create boundaries
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (abs(x[1]) < DOLFIN_EPS or abs(x[1] - H) < DOLFIN_EPS)


class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - L) < DOLFIN_EPS


class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS


class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
circle_ID = 4  # this will mark all the boundary, but it will be overwritten later
circle = AllBoundary()
circle.mark(boundaries, circle_ID)
walls_ID = 1
walls = Walls()
walls.mark(boundaries, walls_ID)
outlet_ID = 2
outlet = Outlet()
outlet.mark(boundaries, outlet_ID)
inlet_ID = 3
inlet = Inlet()
inlet.mark(boundaries, inlet_ID)

# Save
File("cylinder.xml") << mesh
File("cylinder_physical_region.xml") << subdomains
File("cylinder_facet_region.xml") << boundaries
XDMFFile("cylinder.xdmf").write(mesh)
XDMFFile("cylinder_physical_region.xdmf").write(subdomains)
XDMFFile("cylinder_facet_region.xdmf").write(boundaries)
