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
pre_step_length = 4.
after_step_length = 14.
pre_step_height = 3.
after_step_height = 5.

# Create mesh
domain = \
    Rectangle(Point(0., 0.), Point(pre_step_length + after_step_length, after_step_height)) - \
    Rectangle(Point(0., 0.), Point(pre_step_length, after_step_height - pre_step_height))
top_subdomain = Rectangle(Point(0., after_step_height - pre_step_height), Point(pre_step_length + after_step_length, after_step_height))
bottom_subdomain = Rectangle(Point(pre_step_length, 0.), Point(pre_step_length + after_step_length, after_step_height - pre_step_height))
domain.set_subdomain(1, top_subdomain)
domain.set_subdomain(2, bottom_subdomain)
mesh = generate_mesh(domain, 50)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ( \
            (x[0] <= pre_step_length and abs(x[1] - after_step_height + pre_step_height) < DOLFIN_EPS) or \
            (x[1] <= after_step_height - pre_step_height and abs(x[0] - pre_step_length) < DOLFIN_EPS) or \
            (x[0] >= pre_step_length and abs(x[1]) < DOLFIN_EPS) \
        )
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - after_step_height) < DOLFIN_EPS
    
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
inlet = Inlet()
inlet_ID = 1
inlet.mark(boundaries, inlet_ID)
bottom = Bottom()
bottom_ID = 2
bottom.mark(boundaries, bottom_ID)
top = Top()
top_ID = 2
top.mark(boundaries, top_ID)

# Save to xml file
File("backward_facing_step.xml") << mesh
File("backward_facing_step_physical_region.xml") << subdomains
File("backward_facing_step_facet_region.xml") << boundaries
XDMFFile("backward_facing_step.xdmf").write(mesh)
XDMFFile("backward_facing_step_physical_region.xdmf").write(subdomains)
XDMFFile("backward_facing_step_facet_region.xdmf").write(boundaries)
