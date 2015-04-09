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
## @file elliptic_coercive_base.py
#  @brief Implementation of the reduced basis method for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg
import os as os
import shutil
import sys

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveBase
#
# Base class containing the interface of the RB method
# for elliptic coercive problems
class EllipticCoerciveBase:
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V):        
    	# $$ ONLINE DATA STRUCTURES $$ #
    	# 1. Online reduced basis space dimension
        self.N = 0
        # 2. Current parameter
        self.mu = []
        # 3a. Number of terms in the affine expansion
        self.Qa = 0
        self.Qf = 0
        # 3b. Theta multiplicative factors of the affine expansion
        self.theta_a = []
        self.theta_f = []
        # 3c. Reduced order matrices/vectors
        self.red_A = []
        self.red_F = []
        # 4. Residual terms
        self.Cf = []
        self.CC = []
        self.CL = []
        self.LL = []
        self.lnq = []
        # 5. Online solution
        self.uN = 0 # vector of dimension N storing the reduced order solution
        
    	# $$ OFFLINE DATA STRUCTURES $$ #
    	# 1. Maximum reduced basis space dimension
        self.Nmax = 10
        # 2. Parameter ranges and training set
        self.mu_range = []
        self.mu = []
        self.xi_train = []
        # 3c. Matrices/vectors resulting from the truth discretization
        self.A_vec = []
        self.F_vec = []
        # 6. Basis functions matrix
        self.Z = []
        # 7. Truth space, functions and inner products
        self.V = V
        self.dim = self.V.dim()
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        u = self.u
        v = self.v
        scalar = inner(u,v)*dx + inner(grad(u),grad(v))*dx # H^1 inner product
        self.S = assemble(scalar) # H^1 inner product matrix
        l2 = inner(u,v)*dx # L^2 inner product
        self.L2 = assemble(l2) # L^2 inner product matrix
        h1 = inner(grad(u),grad(v))*dx # H^1_0 inner product
        self.H1 = assemble(h1) # H^1_0 inner product matrix
        # 8. Auxiliary functions
        self.snap = Function(self.V) # temporary vector for storage of a truth solution
        self.rb = Function(self.V) # temporary vector for storage of the reduced solution
        self.er = Function(self.V) # temporary vector for storage of the error
        # 9. I/O
        self.snap_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.dual_folder = "dual/"
        self.rb_matrices_folder = "rb_matr/"
        self.pp_folder = "pp/" # post processing
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.rb_matrices_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of this object
    #  @{
    
    ## OFFLINE: set maximum reduced basis space dimension
    def setNmax(self, nmax):
        self.Nmax = nmax
    
    ## OFFLINE: set the range of the parameters
    def setmu_range(self, mu_range):
        self.mu_range = mu_range
    
    ## OFFLINE: set the elements in the training set \eta, from a random uniform distribution
    def setxi_train(self, ntrain):
        ss = "[("
        for i in range(len(self.mu_range)):
            ss += "np.random.uniform(self.mu_range[" + str(i) + "][0],self.mu_range[" + str(i) + "][1])"
            if i < len(self.mu_range)-1:
                ss += ", "
            else:
                ss += ") for _ in range(" + str(ntrain) +")]"
        self.xi_train = eval(ss)

    ## ONLINE: set the current value of the parameter
    def setmu(self, mu ):
        self.mu = mu
    
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve
    def online_solve(self,mu,with_plot=True):
        self.load_red_matrices()
        self.setmu(mu)
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        self.rb_solve()
        sol = self.Z[:, 0]*self.uN[0]
        i=1
        for un in self.uN[1:]:
            sol += self.Z[:, i]*un
            i+=1
        self.rb.vector()[:] = sol
        if with_plot == True:
            plot(self.rb, title = "Reduced solution. mu = " + str(self.mu), interactive = True)
    
    # Perform an online solve (internal)
    def rb_solve(self):
        rb_A = self.aff_assemble(self.red_A, self.theta_a)
        rb_F = self.aff_assemble(self.red_F, self.theta_f)
        if isinstance(rb_A, float) == True:
            uN = rb_F/rb_A
        else:
            uN = np.linalg.solve(rb_A, rb_F)
        self.uN = uN
    
    ## Return an error bound for the current solution
    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
    ## Return the numerator of the error bound for the current solution
    def get_eps2 (self):
        theta_a = self.theta_a
        theta_f = self.theta_f
        CC = self.CC
        CL = self.CL
        LL = self.LL
        Qf = self.Qf
        Qa = self.Qa
        uN = self.uN
        eps2 = 0.0
        if Qf > 1 :
            for qf in range(Qf):
                for qfp in range(Qf):
                    eps2 += theta_f[qf]*theta_f[qfp]*CC[qf,qfp]
        else:
            eps2 += theta_f[0]*theta_f[0]*CC
        if self.N == 1:
            for qf in range(Qf):
                for qa in range(Qa):
                    eps2 += 2.0*theta_f[qf]*uN*theta_a[qa]*CL[0][qf,qa]
    
    
            for qa in range(Qa):
                for qap in range(Qa):
                    eps2 += theta_a[qa]*uN*uN*theta_a[qap]*LL[0,0,qa,qap]
            
        else:
            n=0
            for un in self.uN:
                for qf in range(Qf):
                    for qa in range(Qa):
                        eps2 += 2.0* theta_f[qf]*theta_a[qa]*un*CL[n][qf,qa]
                n += 1

            n = 0
            for un in uN:
                for qa in range(Qa):
                    np = 0
                    for unp in uN:
                        for qap in range(Qa):
                            eps2 += theta_a[qa]*un*theta_a[qap]*unp*LL[n,np,qa,qap]
                        np += 1
                n += 1
        return eps2
    
    ## Compute dual terms
    def compute_dual_terms(self):
        N = self.N
        RBu = Function(self.V)
        riez = Function(self.V)
        riez_ = Function(self.V)

        # CC
        Qf = self.Qf
        Qa = self.Qa
        if self.N == 1 :
    
            self.Cf = self.compute_f_dual()
            if Qf > 1:
                self.CC = np.zeros((Qf,Qf))
                for qf in range(0,Qf):
                    for qfp in range(qf,Qf):
                        self.CC[qf,qfp] = self.compute_scalar(self.Cf[qf],self.Cf[qfp],self.S)
                        if qf != qfp:
                            self.CC[qfp,qf] = self.CC[qf,qfp]
            else:
                self.CC = self.compute_scalar(self.Cf[0],self.Cf[0],self.S)
            np.save(self.dual_folder + "CC", self.CC)
    
            RBu.vector()[:] = self.Z[:, 0]
            
            self.lnq = (self.compute_a_dual(RBu),)
    
            la = Function(self.V)
            lap = Function(self.V)
            self.CL = np.zeros((Qf,Qa))

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[0][:, qa]
                    self.CL[qf,qa] = self.compute_scalar(la,self.Cf[qf],self.S)
            self.CL = (self.CL,)
            np.save(self.dual_folder + "CL", self.CL)
            
            # LL
            self.LL = np.zeros((self.Nmax,self.Nmax,self.Qa,self.Qa))
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[0][:, qa]
                for qap in range(qa,Qa):
                    lap.vector()[:] = self.lnq[0][:, qap]
                    self.LL[0,0,qa,qap] = self.compute_scalar(la,lap,self.S)
                    if qa != qap:
                        self.LL[0,0,qap,qa] = self.LL[0,0,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
        else:
            RBu.vector()[:] = self.Z[:, N-1]
            self.lnq += (self.compute_a_dual(RBu),)
            la = Function(self.V)
            lap = Function(self.V)
            cl = np.zeros((Qf,Qa))

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[N-1][:, qa]
                    cl[qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            self.CL += (cl,)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            n = self.N-1
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[n][:, qa]
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = self.lnq[nn][:, qap]
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
    
    
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    def offline(self):
        print "=============================================================="
        print "=             Offline phase begins                           ="
        print "=============================================================="
        print ""
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        os.makedirs(self.pp_folder)
        self.A_vec = self.assemble_truth_a()
        self.F_vec = self.assemble_truth_f()
        self.theta_a = self.compute_theta_a()
        self.theta_f = self.compute_theta_f()
        self.Qa = len(self.theta_a)
        self.Qf = len(self.theta_f)
        for run in range(self.Nmax):
            print "############################## run = ", run, " ######################################"
            
            A = self.aff_assemble(self.A_vec, self.theta_a)
            F = self.aff_assemble(self.F_vec, self.theta_f)
            
            print "truth solve for mu = ", self.mu
            solve(A, self.snap.vector(), F)
            
            print "update base"
            if self.N == 0:
                self.Z = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
                self.Z /= np.sqrt(np.dot(self.Z[:, 0],self.L2*self.Z[:, 0]))
        
            else:
                self.Z = np.hstack((self.Z,self.snap.vector())) # add new basis functions as column vectors
                self.Z = self.GS()
            self.N += 1
            
            np.save(self.basis_folder + "basis", self.Z)
            print "build rb matrices"
            self.build_rb_matrices()
            self.build_rb_vectors()
            np.save(self.rb_matrices_folder + "red_A", self.red_A)
            np.save(self.rb_matrices_folder + "red_F", self.red_F)
            print "solve rb"
            self.rb_solve()
            
            print "build matrices for error estimation (it may take a while)"
            self.compute_dual_terms()
            
            if self.N < self.Nmax:
                print "find next mu"
                self.greedy()
                self.theta_a = self.compute_theta_a()
                self.theta_f = self.compute_theta_f()
            else:
                self.greedy()

            print ""
    
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        delta_max = 0.0
        count = 1
        for mu in self.xi_train:
            self.setmu(mu)
            self.theta_a = self.compute_theta_a()
            self.theta_f = self.compute_theta_f()
            self.rb_solve()
            delta = self.get_delta()
            if delta > delta_max:
                delta_max = delta
                munew = mu
        print "absolute delta_max = ", delta_max
        if os.path.isfile(self.pp_folder + "delta_max.npy") == True:
            d = np.load(self.pp_folder + "delta_max.npy")
            
            np.save(self.pp_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.pp_folder + "mu_greedy.npy")
            np.save(self.pp_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.pp_folder + "delta_max", delta_max)
            np.save(self.pp_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
    
    ## Assemble the truth affine expansion (matrix)
    def aff_assemble(self, vec, theta_v):
        A_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            A_ += vec[i]*theta_v[i]
        return A_
    
    ## Compute the dual of a
    def compute_a_dual(self, RBu):
        riez = Function(self.V)
        i = 0
        for A in self.A_vec:
            solve (self.S,riez.vector(), A*RBu.vector()*(-1.0))
            if i != 0:
                l = np.hstack((l,riez.vector()))
            else:
                l = np.array(riez.vector()).reshape(-1, 1) # as column vector
                i = 1
        return l
    
    ## Compute the dual of f
    def compute_f_dual(self):
        riez_f = Function(self.V)
        riez = ()
        for i in range(self.Qf):
            solve (self.S, riez_f.vector(), self.F_vec[i])
            riez += (riez_f.copy(True),)
        return riez
    
    ## Perform Gram Schmidt orthonormalization
    def GS(self):
        basis = self.Z
        last = basis.shape[1]-1
        b = basis[:, last].copy()
        for i in range(last):
            proj = np.dot(np.dot(b,self.L2*basis[:, i])/np.dot(basis[:, i],self.L2*basis[:, i]),basis[:, i])
            b = b - proj 
        basis[:, last] = b/np.sqrt(np.dot(b,self.L2*b))
        return basis
        
    ## Assemble the reduced order affine expansion (matrix)
    def build_rb_matrices(self):
        dim = self.dim
        red_A = ()
        i = 0
        for A in self.A_vec:
            A = as_backend_type(A)
            if self.N == 1:
                red_A += (np.dot(self.Z.T,A.mat().getValues(range(dim),range(dim)).dot(self.Z)),)
            else:
                red = np.matrix(np.dot(self.Z.T,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),self.Z))))
                red_A += (red,)
                i += 1
        self.red_A = red_A
    
    ## Assemble the reduced order affine expansion (rhs)
    def build_rb_vectors(self):
        dim = self.dim
        red_F = ()
        i = 0
        for F in self.F_vec:
            F = as_backend_type(F)
            red_f = np.dot(self.Z.T, F.vec().getValues(range(dim)) )
            red_F += (red_f,)
        self.red_F = red_F
    
    ## Auxiliary method to computed the scalar product (v1, M*v2)
    def compute_scalar(self,v1,v2,M):
    	return v1.vector().inner(M*v2.vector())
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    def load_red_matrices(self):
        if not self.red_A and not self.red_F and not self.Z and \
                              not self.CC and not self.CL and \
                              not self.LL: # avoid loading multiple times
            self.red_A = np.load(self.rb_matrices_folder + "red_A.npy")
            self.red_F = np.load(self.rb_matrices_folder + "red_F.npy")
            self.Z = np.load(self.basis_folder + "basis.npy")
            self.CC = np.load(self.dual_folder + "CC.npy")
            self.CL = np.load(self.dual_folder + "CL.npy")
            self.LL = np.load(self.dual_folder + "LL.npy")
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 

    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return the alpha_lower bound.
    # example of implementation:
    #    return 1.0
    def get_alpha_lb(self):
        print "The function get_alpha_lb(self) is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function get_alpha_lb(self)!")

    ## Set theta multiplicative terms of the affine expansion of a.
    # example of implementation:
    #    m1 = self.mu[0]
    #    m2 = self.mu[1]
    #    m3 = self.mu[2]
    #    theta_a0 = m1
    #    theta_a1 = m2
    #    theta_a2 = m1*m2+m3/7.0
    #    return (theta_a0, theta_a1, theta_a2)
    def compute_theta_a(self):
        print "The function compute_theta_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_a()!")
    
    ## Set theta multiplicative terms of the affine expansion of f.
    # example of implementation:
    #    m1 = self.mu[0]
    #    m2 = self.mu[1]
    #    m3 = self.mu[2]
    #    theta_f0 = m1
    #    theta_f1 = m2
    #    theta_f2 = m1*m2+m3/7.0
    #    return (theta_f0, theta_f1, theta_f2)
    def compute_theta_f(self):
        print "The function compute_theta_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function compute_theta_f()!")
        
    ## Set matrices resulting from the truth discretization of a.
    # example of implementation:
    #    a0 = inner(grad(u),grad(v))*dx
    #    A0 = assemble(a0)
    #    return (A0,)
    def assemble_truth_a(self):
        print "The function assemble_truth_a() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_a()!")

    ## Set vectors resulting from the truth discretization of f.
    #    f0 = v*ds(1)
    #    F0 = assemble(f0)
    #    return (F0,)
    def assemble_truth_f(self):
        print "The function compute_truth_f() is problem-specific and needs to be overwritten."
        print "Abort program."
        sys.exit("Plase define function assemble_truth_f()!")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

