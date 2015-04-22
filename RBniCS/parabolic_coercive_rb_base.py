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
## @file parabolic_coercive_rb_base.py
#  @brief Implementation of the reduced basis method for parabolic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from parabolic_coercive_base import *
from elliptic_coercive_rb_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveBase
#
# Base class containing the interface of the RB method
# for parabolic coercive problems
class ParabolicCoerciveRBBase(ParabolicCoerciveBase,EllipticCoerciveRBBase):
# Beware of the diamond problem in multiple inheritance: in python precedence is depth-first and then left-to-right

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call the parent initialization
        ParabolicCoerciveBase.__init__(self, V, bc_list)
        EllipticCoerciveRBBase.__init__(self, V, bc_list)
        
        # TODO il resto del metodo
        # $$ ONLINE DATA STRUCTURES $$ #
        # 4. Residual terms
        self.Cf = []
        self.CC = [] # C_ff
        self.CL = [] # C_fa
        self.LL = [] # C_aa
        self.MM = [] # C_mm
        self.CM = [] # C_fm
        self.LM = [] # C_am
        self.lnq = []
        self.mnq = []
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 9. I/O
        self.name = "RB "
        self.snap_folder = "snapshots/"
        self.basis_folder = "basis/"
        self.dual_folder = "dual/"
        self.red_matrices_folder = "red_matr/"
        self.pp_folder = "pp/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution
    def get_delta(self):
        # TODO il resto del metodo
        eps2 = self.get_eps2()
        alpha = self.get_alpha_lb()
        return np.sqrt(np.abs(eps2)/alpha)
    
    ## Return the numerator of the error bound for the current solution
    def get_eps2 (self):
        # TODO il resto del metodo
        theta_a = self.theta_a
        theta_f = self.theta_f
        Qf = self.Qf
        Qa = self.Qa
        uN = self.uN
        
        eps2 = 0.0
        
        CC = self.CC
        if Qf > 1 :
            for qf in range(Qf):
                for qfp in range(Qf):
                    eps2 += theta_f[qf]*theta_f[qfp]*CC[qf,qfp]
        else:
            eps2 += theta_f[0]*theta_f[0]*CC
        
        CL = self.CL
        LL = self.LL
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
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        # TODO il resto del metodo, ma non dovrebbe servire
        pass
        
    ## Update basis matrix
    def update_basis_matrix(self):
        # TODO il resto del metodo
        if self.N == 0:
            self.Z = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
            self.Z /= np.sqrt(np.dot(self.Z[:, 0], self.S*self.Z[:, 0]))
    
        else:
            self.Z = np.hstack((self.Z, self.snap.vector())) # add new basis functions as column vectors
            self.Z = self.GS()
        self.N += 1
    
    ## Perform Gram Schmidt orthonormalization
    def GS(self):
    # TODO il resto del metodo, ma non serve
        basis = self.Z
        last = basis.shape[1]-1
        b = basis[:, last].copy()
        for i in range(last):
            proj = np.dot(np.dot(b,self.S*basis[:, i])/np.dot(basis[:, i],self.S*basis[:, i]),basis[:, i])
            b = b - proj 
        basis[:, last] = b/np.sqrt(np.dot(b,self.S*b))
        return basis
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
    # TODO il resto del metodo, ma non serve?
        delta_max = -1.0
        munew = None
        for mu in self.xi_train:
            self.setmu(mu)
            self.red_solve(self.N)
            delta = self.get_delta()
            if delta > delta_max:
                delta_max = delta
                munew = mu
        print "absolute delta max = ", delta_max
        if os.path.isfile(self.pp_folder + "delta_max.npy") == True:
            d = np.load(self.pp_folder + "delta_max.npy")
            
            np.save(self.pp_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.pp_folder + "mu_greedy.npy")
            np.save(self.pp_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.pp_folder + "delta_max", delta_max)
            np.save(self.pp_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        
    ## Compute dual terms
    def compute_dual_terms(self):
    # CHECK il resto del metodo
        N = self.N
        RBu = Function(self.V)
        
        Qf = self.Qf
        Qa = self.Qa
        Qm = self.Qm
        if self.N == 1 :
            
            # CC (does not depend on N, so we compute it only once)
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


            self.mnq = (self.compute_m_dual(RBu),)

            lm = Function(self.V)
            lmp = Function(self.V)
            self.CM = np.zeros((Qf,Qm))

            # CM
            for qf in range(0,Qf):
                for qm in range(0,Qm):
                    lm.vector()[:] = self.mnq[0][:, qm]
                    self.CM[qf,qm] = self.compute_scalar(lm,self.Cf[qf],self.S)
            self.CM = (self.CM,)
            np.save(self.dual_folder + "CM", self.CM)

            # MM
            self.MM = np.zeros((self.Nmax,self.Nmax,self.Qm,self.Qm))
            for qm in range(0,Qm):
                lm.vector()[:] = self.mnq[0][:, qm]
                for qmp in range(qm,Qm):
                    lmp.vector()[:] = self.mnq[0][:, qmp]
                    self.MM[0,0,qm,qmp] = self.compute_scalar(lm,lmp,self.S)
                    if qm != qmp:
                        self.MM[0,0,qmp,qm] = self.MM[0,0,qm,qmp]
            np.save(self.dual_folder + "MM", self.MM)

            # LM
            self.LM = np.zeros((self.Nmax,self.Nmax,self.Qm,self.Qa))
            for qm in range(0,Qm):
                lm.vector()[:] = self.mnq[0][:, qm]
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[0][:, qa]
                    self.LM[0,0,qm,qa] = self.compute_scalar(lm,la,self.S)
            np.save(self.dual_folder + "LM", self.LM)
        else:
            n = self.N-1
            RBu.vector()[:] = self.Z[:, n]
            self.lnq += (self.compute_a_dual(RBu),)
            la = Function(self.V)
            lap = Function(self.V)
            cl = np.zeros((Qf,Qa))

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = self.lnq[n][:, qa]
                    cl[qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            self.CL += (cl,)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            for qa in range(0,Qa):
                la.vector()[:] = self.lnq[n][:, qa]
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = self.lnq[nn][:, qap]
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)

            self.mnq += (self.compute_m_dual(RBu),)

            lm = Function(self.V)
            lmp = Function(self.V)
            self.CM = np.zeros((Qf,Qm))

            cm = np.zeros((Qf,Qm))

            # CM
            for qf in range(0,Qf):
                for qm in range(0,Qm):
                    lm.vector()[:] = self.mnq[n][:, qm]
                    cm[qf,qm] = self.compute_scalar(self.Cf[qf],lm,self.S)
            self.CM += (cm,)
            np.save(self.dual_folder + "CM", self.CM)
    
            # MM
            for qm in range(0,Qm):
                lm.vector()[:] = self.mnq[n][:, qm]
                for nn in range(0,N):
                    for qmp in range(0,Qm):
                        lmp.vector()[:] = self.mnq[nn][:, qmp]
                        self.MM[n,nn,qm,qmp] = self.compute_scalar(lm,lmp,self.S)
                        if n != nn:
                            self.MM[nn,n,qm,qmp] = self.MM[n,nn,qm,qmp]
            np.save(self.dual_folder + "MM", self.MM)
    
            # LM
            for qm in range(0,Qm):
                lm.vector()[:] = self.mnq[n][:, qm]
                for nn in range(0,N):
                    for qa in range(0,Qa):
                        la.vector()[:] = self.lnq[nn][:, qa]
                        self.LM[n,nn,qm,qa] = self.compute_scalar(lm,la,self.S)
                        if n != nn:
                            self.LM[nn,n,qm,qa] = self.LM[n,nn,qm,qa]
            np.save(self.dual_folder + "LM", self.LM)


    ## Compute the dual of m
    def compute_m_dual(self, RBu):
        riesz = Function(self.V)
        i = 0
        for M in self.truth_M:
            solve (self.S,riesz.vector(), M*RBu.vector()*(-1.0))
            if i != 0:
                m = np.hstack((m,riesz.vector()))
            else:
                m = np.array(riesz.vector()).reshape(-1, 1) # as column vector
                i = 1
        return m
    
    ## Compute the dual of a
    def compute_a_dual(self, RBu):
        riesz = Function(self.V)
        i = 0
        for A in self.truth_A:
            solve (self.S,riesz.vector(), A*RBu.vector()*(-1.0))
            if i != 0:
                l = np.hstack((l,riesz.vector()))
            else:
                l = np.array(riesz.vector()).reshape(-1, 1) # as column vector
                i = 1
        return l
    
    ## Compute the dual of f
    def compute_f_dual(self):
        riesz = Function(self.V)
        c = ()
        for F in self.truth_F:
            solve (self.S, riesz.vector(), F)
            c += (riesz.copy(True),)
        return c
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_red_matrices(self):
        # Read in data structures as in parents
        # (need to call them explicitly because this method was overridden in both parents)
        ParabolicCoerciveBase.load_red_matrices()
        EllipticCoercivePODBase.load_red_matrices()
    
    #  @}
    ########################### end - I/O - end ###########################  
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    # Nothing to be added in this case
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
