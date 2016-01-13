# Copyright (C) 2015-2016 SISSA mathLab
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
## @file solve_hole.py
#  @brief Example 3: geometrical parametrization
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
import numpy as np
from RBniCS import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: GEOMETRICAL PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Hole(EllipticCoercivePODBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, mesh, subd, bound):
        # Call the standard initialization
        EllipticCoercivePODBase.__init__(self, V, None)
        # ... and also store FEniCS data structures for assembly ...
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        # ... and, finally, FEniCS data structure related to the geometrical parametrization
        self.mesh = mesh
        self.subd = subd
        self.xref = mesh.coordinates()[:,0].copy()
        self.yref = mesh.coordinates()[:,1].copy()
        self.deformation_V = VectorFunctionSpace(self.mesh, "Lagrange", 1)
        self.subdomain_id_to_deformation_dofs = ()
        for subdomain_id in np.unique(self.subd.array()):
            self.subdomain_id_to_deformation_dofs += ([],)
        for cell in cells(mesh):
            subdomain_id = int(self.subd.array()[cell.index()] - 1) # tuple start from 0, while subdomains from 1
            dofs = self.deformation_V.dofmap().cell_dofs(cell.index())
            for dof in dofs:
                self.subdomain_id_to_deformation_dofs[subdomain_id].append(dof)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        m3 = self.mu[2]
        # subdomains 1 and 7
        theta_a0 = - (m2 - 2)/m1 - (2*(2*m1 - 2)*(m1 - 1))/(m1*(m2 - 2)) #K11
        theta_a1 = -m1/(m2 - 2) #K22
        theta_a2 = -(2*(m1 - 1))/(m2 - 2) #K12 and K21
        # subdomains 2 and 8
        theta_a3 = 2 - (m1 - 1)*(m1 - 1)/(m2 - 2) - m2
        theta_a4 = -1/(m2 - 2)
        theta_a5 = (m1 - 1)/(m2 - 2)
        # subdomains 3 and 5
        theta_a6 = -m2/(m1 - 2)
        theta_a7 = - (m1 - 2)/m2 - (2*(2*m2 - 2)*(m2 - 1))/(m2*(m1 - 2))
        theta_a8 = -(2*(m2 - 1))/(m1 - 2)
        # subdomains 4 and 6
        theta_a9 = -1/(m1 - 2)
        theta_a10 = 2 - (m2 - 1)*(m2 - 1)/(m1 - 2) - m1
        theta_a11 = (m2 - 1)/(m1 - 2)
        # boundaries 5, 6, 7 and 8
        theta_a12 = m3
        
        return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8, theta_a9, theta_a10, theta_a11, theta_a12)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        m1 = self.mu[0]
        m2 = self.mu[1]
        theta_f0 = - m1*(m2 - 2.0) # boundary 1
        theta_f1 = - m2*(m1 - 2.0) # boundary 2
        theta_f2 = - m1*(m2 - 2.0) # boundary 3
        theta_f3 = - m2*(m1 - 2.0) # boundary 4
        
        return (theta_f0, theta_f1, theta_f2, theta_f3)
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        ds = self.ds
        # subdomains 1 and 7
        a0 = inner(u.dx(0), v.dx(0))*dx(1) +  inner(u.dx(0), v.dx(0))*dx(7)
        a1 = inner(u.dx(1), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(1))*dx(7)
        a2 = inner(u.dx(0), v.dx(1))*dx(1) +  inner(u.dx(1), v.dx(0))*dx(1) - (inner(u.dx(0), v.dx(1))*dx(7) +  inner(u.dx(1), v.dx(0))*dx(7))
        # subdomains 2 and 8
        a3 = inner(u.dx(0), v.dx(0))*dx(2) +  inner(u.dx(0), v.dx(0))*dx(8)
        a4 = inner(u.dx(1), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(1))*dx(8)
        a5 = inner(u.dx(0), v.dx(1))*dx(2) +  inner(u.dx(1), v.dx(0))*dx(2) - (inner(u.dx(0), v.dx(1))*dx(8) +  inner(u.dx(1), v.dx(0))*dx(8))
        # subdomains 3 and 5
        a6 = inner(u.dx(0), v.dx(0))*dx(3) +  inner(u.dx(0), v.dx(0))*dx(5)
        a7 = inner(u.dx(1), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(1))*dx(5)
        a8 = inner(u.dx(0), v.dx(1))*dx(3) +  inner(u.dx(1), v.dx(0))*dx(3) - (inner(u.dx(0), v.dx(1))*dx(5) +  inner(u.dx(1), v.dx(0))*dx(5))
        # subdomains 4 and 6
        a9 = inner(u.dx(0), v.dx(0))*dx(4) +  inner(u.dx(0), v.dx(0))*dx(6)
        a10 = inner(u.dx(1), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(1))*dx(6)
        a11 = inner(u.dx(0), v.dx(1))*dx(4) +  inner(u.dx(1), v.dx(0))*dx(4) - (inner(u.dx(0), v.dx(1))*dx(6) +  inner(u.dx(1), v.dx(0))*dx(6))
        # boundaries 5, 6, 7 and 8
        a12 = inner(u,v)*ds(5) + inner(u,v)*ds(6) + inner(u,v)*ds(7) + inner(u,v)*ds(8)
        
        # Assemble and return
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
        return (A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        ds = self.ds
        f0 = v*ds(1) # boundary 1
        f1 = v*ds(2) # boundary 2
        f2 = v*ds(3) # boundary 3
        f3 = v*ds(4) # boundary 4
        
        # Assemble and return
        F0 = assemble(f0)
        F1 = assemble(f1)
        F2 = assemble(f2)
        F3 = assemble(f3)
        return (F0, F1, F2, F3)
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve: method overridden to perform
    # the plot on the deformed domain
    def online_solve(self, N=None, with_plot=True):
        # Call the parent method, disabling plot ...
        EllipticCoercivePODBase.online_solve(self, N, False)
        # ... and then deform the mesh and perform the plot
        if with_plot == True:
            self.move_mesh()
            plot(self.red, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
            self.reset_reference()
    
    ## Deform the mesh as a function of the geometrical parameters mu_1 and mu_2
    def move_mesh(self):
        print "moving mesh (it may take a while)"
        displacement = self.compute_displacement()
        ALE.move(self.mesh, displacement)
    
    ## Restore the reference mesh
    def reset_reference(self):
        print "back to the reference mesh"
        new_coor = np.array([self.xref, self.yref]).transpose()
        self.mesh.coordinates()[:] = new_coor
    
    ## Auxiliary method to deform the domain
    def compute_displacement(self):
        expression_displacement_subdomains = (
            Expression(("2.0 - 2.0*mu_1 + mu_1*x[0] - x[0] +(2.0-2.0*mu_1)*x[1]", "2.0 -2.0*mu_2 + (1.0-mu_2)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 1
            Expression(("2.0*mu_1-2.0 +(mu_1-1.0)*x[1]", "2.0 -2.0*mu_2 + (1.0-mu_2)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 2
            Expression(("2.0 - 2.0*mu_1 + (1.0-mu_1)*x[0]", "2.0 -2.0*mu_2 + (2.0-2.0*mu_2)*x[0] + (mu_2 - 1.0)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 3
            Expression(("2.0 - 2.0*mu_1 + (1.0-mu_1)*x[0]", "2.0*mu_2 -2.0 + (mu_2-1.0)*x[0]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 4
            Expression(("2.0*mu_1 -2.0 + (1.0-mu_1)*x[0]", "2.0 -2.0*mu_2 + (2.0*mu_2-2.0)*x[0] + (mu_2 - 1.0)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 5
            Expression(("2.0*mu_1 -2.0 + (1.0-mu_1)*x[0]", "2.0*mu_2 -2.0 + (1.0 - mu_2)*x[0]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 6
            Expression(("2.0 -2.0*mu_1 + (mu_1-1.0)*x[0] + (2.0*mu_1-2.0)*x[1]", "2.0*mu_2 -2.0 + (1.0 - mu_2)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]), # subdomain 7
            Expression(("2.0*mu_1 -2.0 + (1.0-mu_1)*x[1]", "2.0*mu_2 -2.0 + (1.0 - mu_2)*x[1]"), mu_1 = self.mu[0], mu_2 = self.mu[1]) # subdomain 8
        )
        displacement_subdomains = ()
        for i in range(len(expression_displacement_subdomains)):
            displacement_subdomains += (interpolate(expression_displacement_subdomains[i], self.deformation_V),)
        displacement = Function(self.deformation_V)
        for i in range(len(displacement_subdomains)):
            subdomain_dofs = self.subdomain_id_to_deformation_dofs[i]
            displacement.vector()[subdomain_dofs] = displacement_subdomains[i].vector()[subdomain_dofs]
        return displacement
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Export solution in VTK format: add lifting and deform
    def export_solution(self, solution, filename):
        self.move_mesh()
        file = File(filename + ".pvd", "compressed")
        file << solution
        self.reset_reference()
        
    #  @}
    ########################### end - I/O - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 3: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/hole.xml")
subd = MeshFunction("size_t", mesh, "data/hole_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/hole_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Hole class
hole = Hole(V, mesh, subd, bound)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Set mu range, xi_train and Nmax
mu_range = [(1.0, 1.5), (1.0, 1.5), (0.01, 1.0)]
hole.setmu_range(mu_range)
hole.setxi_train(500)
hole.setNmax(20)

# 6. Perform the offline phase
first_mu = (0.5, 0.5, 0.01)
hole.setmu(first_mu)
hole.offline()

# 7. Perform an online solve
online_mu = (0.5,0.5,0.01)
hole.setmu(online_mu)
hole.online_solve()

# 8. Perform an error analysis
hole.setxi_test(500)
hole.error_analysis()
