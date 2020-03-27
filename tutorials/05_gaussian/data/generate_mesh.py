# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Create mesh
rectangle = Rectangle(Point(-1., -1.), Point(1., 1.))
domain = rectangle
mesh = generate_mesh(domain, 20)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)

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
File("gaussian.xml") << mesh
File("gaussian_physical_region.xml") << subdomains
File("gaussian_facet_region.xml") << boundaries
XDMFFile("gaussian.xdmf").write(mesh)
XDMFFile("gaussian_physical_region.xdmf").write(subdomains)
XDMFFile("gaussian_facet_region.xdmf").write(boundaries)
