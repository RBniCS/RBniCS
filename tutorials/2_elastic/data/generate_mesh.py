from dolfin import *
from mshr import *

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
for i in range(3):
    for j in range(3):
        domain.set_subdomain(i + j*3 + 1, Rectangle(Point(i/3., j/3.), Point((i+1)/3., (j+1)/3.)))
mesh = generate_mesh(domain, 32)
plot(mesh)
interactive()

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)
interactive()

# Create boundaries
class Left(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Right(SubDomain):
    def __init__(self, y_min, y_max):
        SubDomain.__init__(self)
        self.y_min = y_min
        self.y_max = y_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS and x[1] >= self.y_min and x[1] <= self.y_max

class Bottom(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
                
class Top(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max
        
boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)
for i in range(3):
    left = Left(i/3., (i+1)/3.)
    left.mark(boundaries, 6)
    top = Top(i/3., (i+1)/3.)
    top.mark(boundaries, 5)
    right = Right(i/3., (i+1)/3.)
    right.mark(boundaries, i+2)
    bottom = Bottom(i/3., (i+1)/3.)
    bottom.mark(boundaries, 1)
plot(boundaries)
interactive()

# Save
File("elastic.xml") << mesh
File("elastic_physical_region.xml") << subdomains
File("elastic_facet_region.xml") << boundaries
