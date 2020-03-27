# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Create mesh
rectangle = Rectangle(Point(-1., -1.), Point(1., 1.))
circle = Circle(Point(0., 0.), 0.5, segments=32)
domain = rectangle
domain.set_subdomain(1, circle)
domain.set_subdomain(2, rectangle - circle)
mesh = generate_mesh(domain, 15)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] + 1.) < DOLFIN_EPS

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] + 1.) < DOLFIN_EPS

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
bottom = Bottom()
bottom.mark(boundaries, 1)
left = Left()
left.mark(boundaries, 2)
right = Right()
right.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)

# Save
File("thermal_block.xml") << mesh
File("thermal_block_physical_region.xml") << subdomains
File("thermal_block_facet_region.xml") << boundaries
XDMFFile("thermal_block.xdmf").write(mesh)
XDMFFile("thermal_block_physical_region.xdmf").write(subdomains)
XDMFFile("thermal_block_facet_region.xdmf").write(boundaries)
