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
## @file solve_graetz.py
#  @brief Example 4: Graetz test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from elliptic_coercive_rb_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: GRAETZ CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Graetz(EllipticCoerciveRBBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, mesh, subd, bound):
        # Call the standard initialization
        EllipticCoerciveRBBase.__init__(self,V)
        # ... and also store FEniCS data structures for assembly ...
        self.dx = Measure("dx")[subd]
        self.ds = Measure("ds")[bound]
        # ... and FEniCS data structure related to the geometrical parametrization
        self.mesh = mesh
        self.subd = subd
        self.xref = mesh.coordinates()[:,0].copy()
        self.yref = mesh.coordinates()[:,1].copy()
        # We will consider non-homogeneous Dirichlet BCs with a lifting.
        # First of all, assemble a suitable lifting function
        lifting_BC = [ 
            DirichletBC(V, 0.0, bound, 1), # homog. bcs
            DirichletBC(V, 0.0, bound, 5), # homog. bcs
            DirichletBC(V, 0.0, bound, 6), # homog. bcs
            DirichletBC(V, 1.0, bound, 2), # non-homog. bcs
            DirichletBC(V, 1.0, bound, 4)  # non-homog. bcs
        ]
        u = self.u
        v = self.v
        dx = self.dx
        lifting_a = inner(grad(u),grad(v))*dx
        lifting_A = assemble(lifting_a)
        lifting_f = 1e-15*v*dx
        lifting_f = assemble(lifting_f)
        [bc.apply(lifting_A) for bc in lifting_BC] # Apply BCs on LHS
        [bc.apply(lifting_F) for bc in lifting_BC] # Apply BCs on RHS
        lifting = Function(V)
        solve(lifting_A, lifting.vector(), lifting_F)
        # Discard the lifting_{BC, A, F} object and store only the lifting function
        self.lifting = lifting
        # Store the BC object for the homogeneous solution (after lifting)
        self.BC = [
            DirichletBC(V, 0.0, bound, 1), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 5), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 6), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 2), # non-homog. bcs with a lifting
            DirichletBC(V, 0.0, bound, 4)  # non-homog. bcs with a lifting
        ]
        # Finally, initialize an SCM object to approximate alpha LB
        self.SCM_obj = SCM(self)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    # Propagate the values of all setters also to the SCM object
    
    def setNmax(self, nmax):
        EllipticCoerciveRBBase.setNmax(nmax)
        self.SCM_obj.setNmax(nmax)
    def settol(self, tol):
        EllipticCoerciveRBBase.settol(tol)
        self.SCM_obj.settol(tol)
    def setmu_range(self, mu_range):
        EllipticCoerciveRBBase.setmu_range(mu_range)
        self.SCM_obj.setmu_range(mu_range)
    def setxi_train(self, ntrain, sampling="random"):
        EllipticCoerciveRBBase.setxi_train(train, sampling)
        self.SCM_obj.setxi_train(train, sampling)
    def setmu(self, mu):
        EllipticCoerciveRBBase.setmu(mu)
        self.SCM_obj.setmu(mu)
        
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return SCM_obj.get_alpha_LB(self.mu)
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_a0 = 10.0
        theta_a1 = 1.0/(mu1*mu2)
        theta_a2 = 1.0/mu2
        theta_a3 = mu1/mu2
        return (theta_a0, theta_a1, theta_a2, theta_a3)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        return (self.mu[2],) # TODO manca il lifting
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        # Define
        vel = Expression("x[1]*(1-x[1])")
        a0 = vel*u.dx(0)*v*dx(1) + vel*u.dx(0)*v*dx(2) + 1e-15*u*v*dx
        a1 = u.dx(0)*v.dx(0)*dx(2) + 1e-15*u*v*dx
        a2 = inner(grad(u),grad(v))*dx(1) + 1e-15*u*v*dx
        a3 = u.dx(1)*v.dx(1)*dx(2) + 1e-15*u*v*dx
        # Assemble
        A0 = assemble(a0)
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)
        # Apply BCs
        [bc.apply(A0) for bc in self.BC]
        [bc.apply(A1) for bc in self.BC]
        [bc.apply(A2) for bc in self.BC]
        [bc.apply(A3) for bc in self.BC]
        [bc.zero(A1) for bc in self.BC]
        [bc.zero(A2) for bc in self.BC]
        [bc.zero(A3) for bc in self.BC]
        # Return
        return (A0, A1, A2, A3)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        ds = self.ds
        # Assemble F0
        f0 = v*ds(2) + v*ds(4) + 1e-11*v*dx
        F0 = assemble(f0)
        # Apply BCs
        [bc.apply(F0) for bc in self.BC]
        # Return
        return (F0,) # TODO manca il lifting
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        # Perform first the SCM offline phase, ...
        self.SCM_obj.offline()
        # ..., and then call the parent method.
        EllipticCoerciveRBBase.offline()
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve: method overridden to perform
    # the plot on the deformed domain
    def online_solve(self,N=None, with_plot=True):
        # Call the parent method, disabling plot ...
        EllipticCoerciveRBBase.online_solve(self, N, False)
        # ... and then deform the mesh and perform the plot
        if with_plot == True:
            self.move_mesh()
            plot(self.red, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
            self.reset_reference()
    
    ## Deform the mesh as a function of the geometrical parameters mu_1 and mu_2
    def move_mesh(self):
        print "moving mesh (it may take a while)"
        x_bar, y_bar = self.reference2deformed()
        new_coor = np.array([x_bar, y_bar]).transpose()
        self.mesh.coordinates()[:] = new_coor
    
    ## Restore the reference mesh
    def reset_reference(self):
        print "back to the refernce mesh"
        new_coor = np.array([self.xref, self.yref]).transpose()
        self.mesh.coordinates()[:] = new_coor
    
    ## Auxiliary method to deform the domain
    def reference2deformed(self):
        subd_v = np.asarray(self.subd.array(), dtype=np.int32)
        m1 = self.mu[0]
        m2 = self.mu[1]
        dd = self.V.dofmap().tabulate_all_coordinates(self.mesh)
        dd.resize((self.V.dim(),2))
        dd_new = dd.copy()
        i=0
        for p in dd_new:
            x = p[0]
            y = p[1]
            
            for c in cells(self.mesh):
                if c.contains(Point(p[0],p[1])):
                    sub_id = subd_v[c.index()]
                    break
            
            # deform coordinates
            if sub_id == 1: 
                p[0] = x # TODO
                p[1] = y # TODO
    
            if sub_id == 2: 
                p[0] = x # TODO
                p[1] = y # TODO
    
        return [dd_new[:,0], dd_new[:,1]]
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 


#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/graetz.xml")
subd = MeshFunction("size_t", mesh, "data/graetz_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/graetz_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the Graetz class
graetz = Graetz(V, mesh, subd, bound)

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Set mu range, xi_train and Nmax
mu_range = [(1.0, 10.0), (0.1, 100.0), (-1.0, 1.0)]
graetz.setmu_range(mu_range)
graetz.setxi_train(500)
graetz.setNmax(4)

# 6. Perform the offline phase
first_mu = (1.0, 1.0, 1.0)
graetz.setmu(first_mu)
graetz.offline()

# 7. Perform an online solve
online_mu = (1.0, 1.0, 1.0)
graetz.setmu(online_mu)
graetz.online_solve()

# 8. Perform an error analysis
graetz.error_analysis()
