# Copyright (C) 2015 SISSA mathLab
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
from elliptic_coercive_base import *

class Hole(EllipticCoerciveBase):

    def __init__(self, V, mesh, subd):

        EllipticCoerciveBase.__init__(self,V)

        self.mesh = mesh
        self.subd = subd
        self.xref = mesh.coordinates()[:,0].copy()
        self.yref = mesh.coordinates()[:,1].copy()

    def get_alpha_lb(self):
        return 1.0/10.0


    def compute_theta_a(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        m3 = self.mu[2]
    # subd 1 & 7
        theta_a0 = abs(m1*(m2 - 2.0))/(m1*m1) + (4.0*abs(m1*(m2 - 2.0))*(m1 - 1.0)*(m1 - 1.0))/(m1*m1*(m2 - 2.0)*(m2 - 2.0)) #K11
        theta_a1 = abs(m1*(m2 - 2.0))/((m2 - 2.0)*(m2 - 2.0)) #K22
        theta_a2 = (2.0*abs(m1*(m2 - 2.0))*(m1 - 1.0))/(m1*(m2 - 2.0)*(m2 - 2.0)) #K12 and K21
    # subd 2 & 8
        theta_a3 = abs(m2 - 2.0) + (abs(m2 - 2.0)*(m1 - 1.0)*(m1 - 1.0))/((m2 - 2.0)*(m2 - 2.0))
        theta_a4 = abs(m2 - 2.0)/((m2 - 2.0)*(m2 - 2.0))
        theta_a5 = -(abs(m2 - 2.0)*(m1 - 1.0))/((m2 - 2.0)*(m2 - 2.0))
    # subd 3 & 5
        theta_a6 = abs(m2*(m1 - 2.0))/((m1 - 2.0)*(m1 - 2.0))
        theta_a7 = abs(m2*(m1 - 2.0))/(m2*m2) + (4.0*abs(m2*(m1 - 2.0))*(m2 - 1.0)*(m2 - 1.0))/(m2*m2*(m1 - 2.0)*(m1 - 2.0))
        theta_a8 = (2.0*abs(m2*(m1 - 2.0))*(m2 - 1.0))/(m2*(m1 - 2.0)*(m1 - 2.0))
    # subd 4 & 6
        theta_a9 = abs(m1 - 2.0)/((m1 - 2.0)*(m1 - 2.0))
        theta_a10 = abs(m1 - 2.0) + (abs(m1 - 2.0)*(m2 - 1.0)*(m2 - 1.0))/((m1 - 2.0)*(m1 - 2.0))
        theta_a11 = -(abs(m1 - 2.0)*(m2 - 1.0))/((m1 - 2.0)*(m1 - 2.0))

        theta_a12 = m3
    
        theta_a = (theta_a0, theta_a1 ,theta_a2 ,theta_a3 ,theta_a4 ,theta_a5 ,theta_a6 ,theta_a7 ,theta_a8 ,theta_a9 ,theta_a10 ,theta_a11, theta_a12)
        self.theta_a = theta_a
    
    def compute_theta_f(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        theta_f0 = abs(m1*(m2 - 2.0))
        theta_f1 = abs(m2*(m1 - 2.0))
        theta_f2 = abs(m1*(m2 - 2.0))
        theta_f3 = abs(m2*(m1 - 2.0))
        self.theta_f = (theta_f0, theta_f1, theta_f2, theta_f3)

    def reference2original(self):
        subd_v = np.asarray(self.subd.array(), dtype=np.int32)
        m1 = self.mu[0]
        m2 = self.mu[1]
        dd = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        dd.resize((self.V.dim(),2))
        dd_new = dd.copy()
        #print x-dd[:,0]
        i=0
        for p in dd_new:
            x = p[0]
            y = p[1]
    
            for c in cells(self.mesh):
                if c.contains(Point(p[0],p[1])):
                   # print c.index()
                   # print "subdomain =", subd_v[c.index()]
                    sub_id = subd_v[c.index()]
                    break
    
            #cambio coordinate
            if sub_id == 1: 
                p[0] = 2.0 - 2.0*m1 + m1*x +(2.0-2.0*m1)*y
                p[1] = 2.0 -2.0*m2 + (2.0-m2)*y
    
            if sub_id == 2: 
                p[0] = 2.0*m1-2.0 + x +(m1-1.0)*y
                p[1] = 2.0 -2.0*m2 + (2.0-m2)*y
    
            if sub_id == 3: 
                p[0] = 2.0 - 2.0*m1 + (2.0-m1)*x
                p[1] = 2.0 -2.0*m2 + (2.0-2.0*m2)*x +m2*y
    
            if sub_id == 4: 
                p[0] = 2.0 - 2.0*m1 + (2.0-m1)*x
                p[1] = 2.0*m2 -2.0 + (m2-1.0)*x + y
    
            if sub_id == 5: 
                p[0] = 2.0*m1 -2.0 + (2.0-m1)*x
                p[1] = 2.0 -2.0*m2 + (2.0*m2-2.0)*x + m2*y
    
            if sub_id == 6: 
                p[0] = 2.0*m1 -2.0 + (2.0-m1)*x
                p[1] = 2.0*m2 -2.0 + (1.0 - m2)*x + y
    
            if sub_id == 7: 
                p[0] = 2.0 -2.0*m1 + m1*x + (2.0*m1-2.0)*y
                p[1] = 2.0*m2 -2.0 + (2.0 - m2)*y
    
            if sub_id == 8: 
                p[0] = 2.0*m1 -2.0 + x + (1.0-m1)*y
                p[1] = 2.0*m2 -2.0 + (2.0 - m2)*y
    
        return [dd_new[:,0], dd_new[:,1]]

    def move_mesh(self):
        print "moving mesh (it may take a while)"
        x_bar, y_bar = self.reference2original()
        new_coor = np.array([x_bar, y_bar]).transpose()
        self.mesh.coordinates()[:] = new_coor

    def reset_reference(self):
        print "back to the refernce mesh"
        new_coor = np.array([self.xref, self.yref]).transpose()
        self.mesh.coordinates()[:] = new_coor

    def online_solve(self,mu):
        self.load_red_matrices()
        self.setmu(mu)
        self.compute_theta_a()
        self.compute_theta_f()
        self.rb_solve()
        sol = self.Z[0]*self.uN[0]
        i=1
        for un in self.uN[1:]:
            sol += self.Z[i]*un
            i+=1
        self.rb.vector()[:] = sol


        self.move_mesh()
        File("rb.pvd") << self.rb
        plot(self.rb, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
        self.reset_reference()




parameters.linear_algebra_backend = 'PETSc'

# Create mesh and define function space
mesh = Mesh("hole.xml")
subd = MeshFunction("size_t", mesh, "hole_physical_region.xml")
bound = MeshFunction("size_t", mesh, "hole_facet_region.xml")


x = mesh.coordinates()[:,0].copy()
y = mesh.coordinates()[:,1].copy()

V = FunctionSpace(mesh, "Lagrange", 1)

hole = Hole(V,mesh,subd)


# Define new measures associated with the interior domains and
# exterior boundaries
dx = Measure("dx")[subd]
ds = Measure("ds")[bound]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# subd 1 & 7
a0 = inner(u.dx(0), v.dx(0))*dx(1) +  inner(u.dx(0), v.dx(0))*dx(7)
a1 = inner(u.dx(1), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(1))*dx(7)
a2 = inner(u.dx(0), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(0))*dx(1) - (inner(u.dx(0), v.dx(1))*dx(7) +  inner(u.dx(1), v.dx(0))*dx(7))
# subd 2 & 8
a3 = inner(u.dx(0), v.dx(0))*dx(2) +  inner(u.dx(0), v.dx(0))*dx(8)
a4 = inner(u.dx(1), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(1))*dx(8)
a5 = inner(u.dx(0), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(0))*dx(2) - (inner(u.dx(0), v.dx(1))*dx(8) +  inner(u.dx(1), v.dx(0))*dx(8))
# subd 3 & 5
a6 = inner(u.dx(0), v.dx(0))*dx(3) +  inner(u.dx(0), v.dx(0))*dx(5)
a7 = inner(u.dx(1), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(1))*dx(5)
a8 = inner(u.dx(0), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(0))*dx(3) - (inner(u.dx(0), v.dx(1))*dx(5) +  inner(u.dx(1), v.dx(0))*dx(5))
# subd 4 & 6
a9 = inner(u.dx(0), v.dx(0))*dx(4) +  inner(u.dx(0), v.dx(0))*dx(6)
a10 = inner(u.dx(1), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(1))*dx(6)
a11 = inner(u.dx(0), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(0))*dx(4) - (inner(u.dx(0), v.dx(1))*dx(6) +  inner(u.dx(1), v.dx(0))*dx(6))
a12 = inner(u,v)*ds(5) + inner(u,v)*ds(6) + inner(u,v)*ds(7) + inner(u,v)*ds(8)  

f0 = v*ds(1)
f1 = v*ds(2)
f2 = v*ds(3)
f3 = v*ds(4)
#out0 = f0 + f1 + f2


A0 = assemble(a0)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)
A5 = assemble(a5)
A6 = assemble(a6)
A7 = assemble(a7)
A8 = assemble(a8)
A9 = assemble(a9)
A10 = assemble(a10)
A11 = assemble(a11)
A12 = assemble(a12)

F0 = assemble(f0)
F1 = assemble(f1)
F2 = assemble(f2)
F3 = assemble(f3)

#Out0 = assemble(out0)



A_vec = (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
hole.setA_vec(A_vec)

F_vec = (F0, F1, F2, F3)
hole.setF_vec(F_vec)


mu_range = [(0.5,1.5), (0.5, 1.5), (0.01,1.0)]
hole.setmu_range(mu_range)
hole.settheta_train(3000)
first_mu = (0.5, 0.5, 0.01)
hole.setNmax(3)

hole.setmu(first_mu)
hole.offline()
hole.online_solve((0.5,0.5,0.01))
