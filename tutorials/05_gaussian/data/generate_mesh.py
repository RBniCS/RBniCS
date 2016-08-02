from dolfin import *
from mshr import *

coarse = False
if coarse:
    n = 10
    filename = "gaussian_coarse"
else:
    n = 20
    filename = "gaussian"

# Create mesh
rectangle = Rectangle(Point(-1., -1.), Point(1., 1.))
domain = rectangle
mesh = generate_mesh(domain, n)
plot(mesh)
interactive()

# Create subdomains
subdomains = CellFunction("size_t", mesh)
subdomains.set_all(0)
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
File(filename + ".xml") << mesh
File(filename + "_physical_region.xml") << subdomains
File(filename + "_facet_region.xml") << boundaries
