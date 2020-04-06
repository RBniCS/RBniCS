# Copyright (C) 2015-2020 by the RBniCS authors
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
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary = Boundary()
boundary.mark(boundaries, 1)

# Save
File("square.xml") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries
XDMFFile("square.xdmf").write(mesh)
XDMFFile("square_physical_region.xdmf").write(subdomains)
XDMFFile("square_facet_region.xdmf").write(boundaries)
