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

# Geometrical parameters
t = 1.
D = 1.
L = 1.
S = 1.
H = 1.

# Create mesh
subdomain_1 = Rectangle(Point(D, L), Point(D + H, L + t))
subdomain_2 = Rectangle(Point(0., L + t), Point(D, L + t + S))
subdomain_3 = Rectangle(Point(0., L), Point(D, L + t))
subdomain_4 = Rectangle(Point(0., 0.), Point(D, L))
domain = subdomain_1 + subdomain_2 + subdomain_3 + subdomain_4
domain.set_subdomain(1, subdomain_1)
domain.set_subdomain(2, subdomain_2)
domain.set_subdomain(3, subdomain_3)
domain.set_subdomain(4, subdomain_4)
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
        return on_boundary and ( \
            abs(x[0]) < DOLFIN_EPS or \
            abs(x[1] - D - t - S) < DOLFIN_EPS or \
            ((x[1] <= L or x[1] >= L + t) and abs(x[0] - D) < DOLFIN_EPS) or \
            (x[0] >= D and (abs(x[1] - L) < DOLFIN_EPS or abs(x[1] - L - t) < DOLFIN_EPS))
        )
        
boundaries = FacetFunction("size_t", mesh)
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
File("t_bypass.xml") << mesh
File("t_bypass_physical_region.xml") << subdomains
File("t_bypass_facet_region.xml") << boundaries
XDMFFile("t_bypass.xdmf").write(mesh)
XDMFFile("t_bypass_physical_region.xdmf").write(subdomains)
XDMFFile("t_bypass_facet_region.xdmf").write(boundaries)
