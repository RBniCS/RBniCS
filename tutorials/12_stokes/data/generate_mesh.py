# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *
from rbnics.backends.dolfin.wrapping import counterclockwise
from rbnics.shape_parametrization.utils.symbolic import VerticesMappingIO

# Geometrical parameters
t = 1.  # mu[0]
D = 1.  # mu[1]
L = 1.  # mu[2]
S = 1.  # mu[3]
H = 1.  # mu[4]
theta = pi  # mu[5]

# Define domain
rectangle_1 = Rectangle(Point(D, L), Point(D + H, L + t))
rectangle_2 = Rectangle(Point(0., L + t), Point(D, L + t + S))
rectangle_3 = Rectangle(Point(0., L), Point(D, L + t))
rectangle_4 = Rectangle(Point(0., 0.), Point(D, L))
domain = rectangle_1 + rectangle_2 + rectangle_3 + rectangle_4

# Define vertices mappings of affine shape parametrization. These will be used
# to partition the mesh in subdomains.
vertices_mappings = [
    {
        ("1.0", "2.0"): ("mu[1]", "mu[2]+mu[0]"),
        ("1.0", "1.0"): ("mu[1]", "mu[2]"),
        ("2.0", "1.0"): ("mu[1]+mu[4]", "mu[2]+(mu[4]*tan(mu[5]))")
    },  # subdomain 1
    {
        ("2.0", "1.0"): ("mu[1]+mu[4]", "mu[2]+(mu[4]*tan(mu[5]))"),
        ("2.0", "2.0"): ("mu[1]+mu[4]", "mu[2]+(mu[4]*tan(mu[5]))+mu[0]"),
        ("1.0", "2.0"): ("mu[1]", "mu[2]+mu[0]")
    },  # subdomain 2
    {
        ("0.0", "3.0"): ("0.0", "mu[2]+ mu[3] +mu[0]"),
        ("0.0", "2.0"): ("0.0", "mu[2]+mu[0]"),
        ("1.0", "2.0"): ("mu[1]", "mu[2]+mu[0]")
    },  # subdomain 3
    {
        ("1.0", "2.0"): ("mu[1]", "mu[2]+mu[0]"),
        ("1.0", "3.0"): ("mu[1]", "mu[2]+ mu[3] +mu[0]"),
        ("0.0", "3.0"): ("0.0", "mu[2]+ mu[3] +mu[0]")
    },  # subdomain 4
    {
        ("0.0", "2.0"): ("0.0", "mu[2]+mu[0]"),
        ("0.0", "1.0"): ("0.0", "mu[2]"),
        ("1.0", "1.0"): ("mu[1]", "mu[2]")
    },  # subdomain 5
    {
        ("1.0", "1.0"): ("mu[1]", "mu[2]"),
        ("1.0", "2.0"): ("mu[1]", "mu[2]+mu[0]"),
        ("0.0", "2.0"): ("0.0", "mu[2]+mu[0]")
    },  # subdomain 6
    {
        ("0.0", "1.0"): ("0.0", "mu[2]"),
        ("0.0", "0.0"): ("0.0", "0.0"),
        ("1.0", "0.0"): ("mu[1]", "0.0")
    },  # subdomain 7
    {
        ("1.0", "0.0"): ("mu[1]", "0.0"),
        ("1.0", "1.0"): ("mu[1]", "mu[2]"),
        ("0.0", "1.0"): ("0.0", "mu[2]")
    }  # subdomain 8
]

# Create mesh
for i, vertices_mapping in enumerate(vertices_mappings):
    subdomain_i = Polygon([Point(*[float(coord) for coord in vertex])
                           for vertex in counterclockwise(vertices_mapping.keys())])
    domain.set_subdomain(i + 1, subdomain_i)
mesh = generate_mesh(domain, 50)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())


# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - D - H) < DOLFIN_EPS


class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < DOLFIN_EPS


class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            abs(x[0]) < DOLFIN_EPS or
            abs(x[1] - D - t - S) < DOLFIN_EPS or
            ((x[1] <= L or x[1] >= L + t) and abs(x[0] - D) < DOLFIN_EPS) or
            (x[0] >= D and (abs(x[1] - L) < DOLFIN_EPS or abs(x[1] - L - t) < DOLFIN_EPS))
        )


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
inlet = Inlet()
inlet_ID = 1
inlet.mark(boundaries, inlet_ID)
outlet = Outlet()
outlet_ID = 2
outlet.mark(boundaries, outlet_ID)
walls = Walls()
walls_ID = 3
walls.mark(boundaries, walls_ID)

# Save to xml file
VerticesMappingIO.save_file(vertices_mappings, ".", "t_bypass_vertices_mapping.vmp")
File("t_bypass.xml") << mesh
File("t_bypass_physical_region.xml") << subdomains
File("t_bypass_facet_region.xml") << boundaries
XDMFFile("t_bypass.xdmf").write(mesh)
XDMFFile("t_bypass_physical_region.xdmf").write(subdomains)
XDMFFile("t_bypass_facet_region.xdmf").write(boundaries)
