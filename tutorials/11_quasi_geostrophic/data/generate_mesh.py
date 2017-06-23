from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 30)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)
plot(subdomains)
interactive()

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
        
boundaries = FacetFunction("size_t", mesh)

bottom = Bottom()
bottom.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 2)
top = Top()
top.mark(boundaries, 3)
left = Left()
left.mark(boundaries, 4)
plot(boundaries)
interactive()

# Save
File("square.xml") << mesh
File("square_physical_region.xml") << subdomains
File("square_facet_region.xml") << boundaries
