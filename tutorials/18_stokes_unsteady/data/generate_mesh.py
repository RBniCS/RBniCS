# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0.0, 0.0), Point(1.0, 1.0))
subdomain1 = Rectangle(Point(0.0, 0.0), Point(0.5, 1.0))
subdomain2 = Rectangle(Point(0.5, 0.0), Point(1.0, 1.0))
domain.set_subdomain(1, subdomain1)  # add some fake subdomains to make sure that the mesh is split
domain.set_subdomain(2, subdomain2)  # at x[0] = 0.5, since we will pin the pressure there
mesh = generate_mesh(domain, 27)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(1)


# Create boundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS


class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < DOLFIN_EPS


class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top = Top()
top.mark(boundaries, 1)
bottom = Bottom()
bottom.mark(boundaries, 2)
left = Left()
left.mark(boundaries, 2)
right = Right()
right.mark(boundaries, 2)

# Save
File("cavity.xml") << mesh
File("cavity_physical_region.xml") << subdomains
File("cavity_facet_region.xml") << boundaries
XDMFFile("cavity.xdmf").write(mesh)
XDMFFile("cavity_physical_region.xdmf").write(subdomains)
XDMFFile("cavity_facet_region.xdmf").write(boundaries)
