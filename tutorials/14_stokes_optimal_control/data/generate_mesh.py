# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

def generate_mesh1():
    # Create mesh
    domain = Rectangle(Point(0., 0.), Point(1., 1.))
    mesh = generate_mesh(domain, 24)

    # Create subdomains
    subdomains = MeshFunction("size_t", mesh, 2)
    subdomains.set_all(1)

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
    boundaries.set_all(0)
    left = Left()
    left.mark(boundaries, 1)
    bottom = Bottom()
    bottom.mark(boundaries, 1)
    top = Top()
    top.mark(boundaries, 1)
    right = Right()
    right.mark(boundaries, 2)

    # Save
    File("mesh1.xml") << mesh
    File("mesh1_physical_region.xml") << subdomains
    File("mesh1_facet_region.xml") << boundaries
    XDMFFile("mesh1.xdmf").write(mesh)
    XDMFFile("mesh1_physical_region.xdmf").write(subdomains)
    XDMFFile("mesh1_facet_region.xdmf").write(boundaries)

generate_mesh1()
