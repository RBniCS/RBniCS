from dolfin import *
from mshr import *

# Create mesh
rectangle = Rectangle(Point(-1., -1.), Point(1., 1.))
circle = Circle(Point(0., 0.), 0.5, segments=32)
domain = rectangle
domain.set_subdomain(1, circle)
domain.set_subdomain(2, rectangle - circle)
mesh = generate_mesh(domain, 15)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)
interactive()

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
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
bottom = Bottom()
bottom.mark(boundaries, 1)
left = Left()
left.mark(boundaries, 2)
right = Right()
right.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
plot(boundaries)
interactive()

# Save
File("tblock.xml") << mesh
File("tblock_physical_region.xml") << subdomains
File("tblock_facet_region.xml") << boundaries
