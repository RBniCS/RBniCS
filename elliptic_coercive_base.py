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
import numpy as np
from scipy.sparse import csr_matrix
import scipy.linalg
import os as os
import shutil
import sys

class EllipticCoerciveBase:

    def __init__(self, V):
        self.N = 0
        self.Nmax = 10
        self.theta_train = []
        self.A_vec = []
        self.Qa = 0
        self.F_vec = []
        self.Qf = 0
        self.mu_range = []
        self.mu = []
        self.Cf = []
        self.CC = []
        self.CL = []
        self.LL = []
        self.lnq = []
        self.theta_a = []
        self.theta_f = []
        self.Z = []
        self.red_A = []
        self.red_F = []
        self.uN = 0
        self.V = V
        self.dim = self.V.dim()
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        scalar = inner(u,v)*dx + inner(grad(u),grad(v))*dx
        self.S = assemble(scalar)
        l2 = inner(u,v)*dx
        L2 = assemble(l2)
        self.L2 = as_backend_type(L2)
        
        h1 = inner(grad(u),grad(v))*dx
        self.H1 = PETScMatrix()
        assemble(h1, tensor = self.H1)
        
        self.snap = Function(self.V)
        self.rb = Function(self.V)
        self.er = Function(self.V)
        self.snap_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.dual_folder = "dual/"
        self.rb_matrices_folder = "rb_matr/"
        self.pp_folder = "pp/" # post processing
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.rb_matrices_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)

    def setNmax(self, nmax):
        self.Nmax = nmax

    def setA_vec(self, avec):
        self.A_vec = avec
        self.Qa = len(self.A_vec)

    def setF_vec(self, fvec):
        self.F_vec = fvec
        self.Qf = len(self.F_vec)

    def settheta_train(self, ntrain):
        ss = "[("
        for i in range(len(self.mu_range)):
            ss += "np.random.uniform(self.mu_range[" + str(i) + "][0],self.mu_range[" + str(i) + "][1])"
            if i < len(self.mu_range)-1:
                ss += ", "
            else:
                ss += ") for _ in range(" + str(ntrain) +")]"
        self.theta_train = eval(ss)

    def setmu(self, mu ):
        self.mu = mu

    def setmu_range(self, mu_range ):
        self.mu_range = mu_range

    def GS(self):
        basis = self.Z
        last = len(basis)-1
        b = basis[last].copy()
        for i in range(last):
            proj = np.dot(np.dot(b,self.L2*basis[i])/np.dot(basis[i],self.L2*basis[i]),basis[i])
            b = b - proj 
        basis[last] = b/np.sqrt(np.dot(b,self.L2*b))
        return basis

    def compute_scalar(self,v1,v2,M):
        return np.dot(v1.vector(),M*v2.vector() )
    
    
    def aff_assemble(self, vec, theta_v):
        A_ = vec[0]*theta_v[0]
        for i in range(1,len(vec)):
            A_ += vec[i]*theta_v[i]
        return A_

    def greedy(self):
        delta_max = 0.0
        count = 1
        for mu in self.theta_train:
            self.setmu(mu)
            self.compute_theta_a()
            self.compute_theta_f()
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
#        return munew

    def get_delta(self):
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
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

    def build_rb_matrices(self):
        dim = self.dim
        red_A = ()
        i = 0
        for A in self.A_vec:
            A = as_backend_type(A)
            if self.N == 1:
                red_A += (np.dot(A.mat().getValues(range(dim),range(dim)).dot(self.Z.T),self.Z),)
            else:
                red = np.matrix(np.dot(self.Z,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),self.Z.T))))
                red_A += (red,)
                i += 1
        self.red_A = red_A
    
    def build_rb_vectors(self):
        dim = self.dim
        red_F = ()
        i = 0
        for F in self.F_vec:
            F = as_backend_type(F)
            red_f = np.dot(self.Z, F.vec().getValues(range(dim)) )
            red_F += (red_f,)
        self.red_F = red_F
    
    def compute_a_dual(self, RBu):
        riez = Function(self.V)
        i = 0
        for A in self.A_vec:
            solve (self.S,riez.vector(), A*RBu.vector()*(-1.0))
            if i != 0:
                l = np.vstack((l,riez.vector()))
            else:
                l = np.array(riez.vector())
                i = 1
        return l
    def compute_f_dual(self):
        riez_f = Function(self.V)
        riez = ()
        for i in range(self.Qf):
            solve (self.S, riez_f.vector(), self.F_vec[i])
            riez += (riez_f.copy(True),)
        return riez
    
    
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
    
            RBu.vector()[:] = self.Z
            
            self.lnq = (self.compute_a_dual(RBu),)
    
            la = Function(self.V)
            lap = Function(self.V)
            self.CL = np.zeros((Qf,Qa))

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[0][qa]
                    self.CL[qf,qa] = self.compute_scalar(la,self.Cf[qf],self.S)
            self.CL = (self.CL,)
            np.save(self.dual_folder + "CL", self.CL)
            
            # LL
            self.LL = np.zeros((self.Nmax,self.Nmax,self.Qa,self.Qa))
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[0][qa]
                for qap in range(qa,Qa):
                    lap.vector()[:] = self.lnq[0][qap]
                    self.LL[0,0,qa,qap] = self.compute_scalar(la,lap,self.S)
                    if qa != qap:
                        self.LL[0,0,qap,qa] = self.LL[0,0,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
        else:
            RBu.vector()[:] = self.Z[N-1]
            self.lnq += (self.compute_a_dual(RBu),)
            la = Function(self.V)
            lap = Function(self.V)
            cl = np.zeros((Qf,Qa))

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[N-1][qa]
                    cl[qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            self.CL += (cl,)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            n = self.N-1
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[n][qa]
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = self.lnq[nn][qap]
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)
    
    
    def rb_solve(self):
        rb_A = self.aff_assemble(self.red_A, self.theta_a)
        rb_F = self.aff_assemble(self.red_F, self.theta_f)
        if isinstance(rb_A, float) == True:
            uN = rb_F/rb_A
        else:
            uN = np.linalg.solve(rb_A, rb_F)
        self.uN = uN

    def offline(self):
        print "=============================================================="
        print "=             Offline phase begins                           ="
        print "=============================================================="
        print ""
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        os.makedirs(self.pp_folder)
        self.compute_theta_a()
        self.compute_theta_f()
        for run in range(self.Nmax):
            print "############################## run = ", run, " ######################################"
            
            A = self.aff_assemble(self.A_vec, self.theta_a)
            F = self.aff_assemble(self.F_vec, self.theta_f)
            
            print "truth solve for mu = ", self.mu
            solve(A, self.snap.vector(), F)
            
            print "update base"
            if self.N == 0:
                self.Z = np.array(self.snap.vector())
                self.Z /= np.sqrt(np.dot(self.Z,self.L2*self.Z))
        
            else:
                self.Z = np.vstack((self.Z,self.snap.vector()))
                self.Z = self.GS()
            self.N += 1
            
            np.save(self.basis_folder + "basis", self.Z)
            print "build_rb matrices"
            self.build_rb_matrices()
            self.build_rb_vectors()
            np.save(self.rb_matrices_folder + "red_A", self.red_A)
            np.save(self.rb_matrices_folder + "red_F", self.red_F)
            print "solve-rb"
            self.rb_solve()
            
            print "build matrices for error estimation (it may take a while)"
            self.compute_dual_terms()
            
            if self.N < self.Nmax:
                print "find next mu"
                self.greedy()
                self.compute_theta_a()
                self.compute_theta_f()
            else:
                self.greedy()

            print ""

    def load_red_matrices(self):
        self.red_A = np.load(self.rb_matrices_folder + "red_A.npy")
        self.red_F = np.load(self.rb_matrices_folder + "red_F.npy")
        self.Z = np.load(self.basis_folder + "basis.npy")
        self.CC = np.load(self.dual_folder + "CC.npy")
        self.CL = np.load(self.dual_folder + "CL.npy")
        self.LL = np.load(self.dual_folder + "LL.npy")

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
        plot(self.rb, title = "Reduced solution. mu = " + str(self.mu), interactive = True)


################# problem specific
    def get_alpha_lb(self):
        """ return the alpha_lower bound.
        example of implementation:
        return 1.0
        """
        print "The function get_alpha_lb(self) is problem-specific and need to be overwritten."
        print "Abort program."
        sys.exit("Define function get_alpha_lb(self)!")


    def compute_theta_a(self):
        """ Set the self.theta_a attribute.

        It must match the length of self.A_vec

        example of implementation:
        m1 = self.mu[0]
        m2 = self.mu[1]
        m3 = self.mu[2]
        theta_a0 = m1
        theta_a1 = m2
        theta_a2 = m1*m2+m3/7.0
        self.theta_a = (theta_a0, theta_a1, theta_a2)
        """
        print "The function compute_theta_a() is problem-specific and need to be overwritten."
        print "Abort program."
        sys.exit("Define function compute_theta_a()!")
    
    def compute_theta_f(self):
        """ Set the self.theta_f attribute.

        It must match the length of self.F_vec

        example of implementation:
        m1 = self.mu[0]
        m2 = self.mu[1]
        m3 = self.mu[2]
        theta_f0 = m1
        theta_f1 = m2
        theta_f2 = m1*m2+m3/7.0
        self.theta_f = (theta_f0, theta_f1, theta_f2)
        """
        print "The function compute_theta_f() is problem-specific and need to be overwritten."
        print "Abort program."
        sys.exit("Define function compute_theta_f()!")

