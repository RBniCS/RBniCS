# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 30)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)


# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
bottom = Bottom()
bottom.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
left = Left()
left.mark(boundaries, 4)

# Save
File("square.xml") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries
XDMFFile("square.xdmf").write(mesh)
XDMFFile("square_physical_region.xdmf").write(subdomains)
XDMFFile("square_facet_region.xdmf").write(boundaries)
