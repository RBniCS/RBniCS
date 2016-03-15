# Copyright (C) 2015-2016 by the RBniCS authors
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
from proper_orthogonal_decomposition import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARABOLIC COERCIVE RB BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParabolicCoerciveBase
#
# Base class containing the interface of the RB method
# for parabolic coercive problems
class ParabolicCoerciveRBBase(ParabolicCoerciveBase,EllipticCoerciveRBBase):
# Beware of the diamond problem in multiple inheritance: in python precedence is depth-first and then left-to-right
    """This class implements the Certified Reduced Basis Method for
    parabolic coercive problems, assuming the compliance of the output
    of interest. It combines a POD approach on the time, with a greedy
    exploration of the parameter space.

    The strategy is the following:

   1. Solve the transients with a given mu. At each time step a
      snapshot will be stored.

   2. A POD is performed on the snapshots just stored. Then, Just few
      of the computed POD modes are retained, according to the
      settings provided by the user (the user can choose if the number
      of POD modes is fixed, or it must satisfy a given tolerance.

   3. The reduced space is enriched with the retained POD modes.

   4. The next parameter is computed with a greed algorithm like the
      RB method for elliptic case, whit the difference that the a
      posteriori error estimation is formulated for the parabolic
      problem. Since the error estimator increases with the time, the
      greedy uses the error bound computed at the LAST time step.

    """

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced basis object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, bc_list):
        # Call the parent initialization
        ParabolicCoerciveBase.__init__(self, V, bc_list)
        EllipticCoerciveRBBase.__init__(self, V, bc_list)

        self.POD = ProperOrthogonalDecomposition()
        self.M1 = 2
        self.M2 = 2
        
        # CHECK il resto del metodo
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
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution
    def get_delta(self): 
        # CHECK il resto del metodo
        alpha = self.get_alpha_lb()
        all_eps2 = self.get_all_eps2()
        delta = np.sqrt(np.abs(np.sum(all_eps2))*self.dt/alpha)
        return delta
    
    def get_all_eps2(self):
        # CHECK
        all_eps2 = np.zeros(len(self.all_times))
        for i in range(len(self.all_times)):
            all_eps2[i] += self.get_eps2(i)
        return all_eps2

    ## Return the numerator of the error bound for the current solution
    def get_eps2(self, tt):
        # CHECK il resto del metodo
        theta_m = self.theta_m
        theta_a = self.theta_a
        theta_f = self.theta_f
        Qf = self.Qf
        Qa = self.Qa
        Qm = self.Qm

        uN = self.all_uN[:, tt]
        if (tt == 0):
            uN_k = uN*0.0
        else:
            uN_k = self.all_uN[:, tt-1]
        
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
        LM = self.LM
        CM = self.CM
        MM = self.MM

        if self.N == 1:
    
            #LL
            for qa in range(Qa):
                for qap in range(Qa):
                    eps2 += theta_a[qa]*uN*uN*theta_a[qap]*LL[0,0,qa,qap]
            #CL
            for qf in range(Qf):
                for qa in range(Qa):
                    eps2 += 2.0*theta_f[qf]*uN*theta_a[qa]*CL[0,qf,qa]
    
            #CM
            for qf in range(Qf):
                for qm in range(Qm):
                    eps2 += (2.0/self.dt)*theta_f[qf]*(uN-uN_k)*theta_m[qm]*CM[0,qf,qm]

            #LM
            for qm in range(Qm):
                for qa in range(Qa):
                    eps2 += (2.0/self.dt)*theta_a[qa]*(uN-uN_k)*uN*theta_m[qm]*LM[0,0,qm,qa]

            #MM
            for qm in range(Qm):
                for qmp in range(Qm):
                    eps2 += (1.0/(self.dt**2))*(uN-uN_k)*(uN-uN_k)*theta_m[qm]*theta_m[qmp]*MM[0,0,qm,qmp]
            
        else:
            #CL
            n=0
            for un in uN:
                for qf in range(Qf):
                    for qa in range(Qa):
                        eps2 += 2.0* theta_f[qf]*theta_a[qa]*un*CL[n,qf,qa]
                n += 1

            #LL
            n = 0
            for un in uN:
                for qa in range(Qa):
                    np = 0
                    for unp in uN:
                        for qap in range(Qa):
                            eps2 += theta_a[qa]*un*theta_a[qap]*unp*LL[n,np,qa,qap]
                        np += 1
                n += 1

            #CM
            for i in range(self.N):
                for qf in range(Qf):
                    for qm in range(Qm):
                        eps2 += (2.0/self.dt)*(uN[i]-uN_k[i])*theta_f[qf]*theta_m[qm]*CM[i,qf,qm]

            #LM
            for i in range(self.N):
                for j in range(self.N):
                    for qm in range(Qm):
                        for qa in range(Qa):
                            eps2 += (2.0/self.dt)*(uN[i]-uN_k[i])*uN[j]*theta_m[qm]*theta_a[qa]*LM[i,j,qm,qa]

            #MM
            for i in range(self.N):
                for j in range(self.N):
                    for qm in range(Qm):
                        for qmp in range(Qm):
                            eps2 += (1.0/(self.dt**2))*(uN[i]-uN_k[i])*(uN[j]-uN_k[j])*theta_m[qm]*theta_m[qmp]*MM[i,j,qm,qm]
        
        return eps2
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
        
    ## Update basis matrix
    def update_basis_matrix(self): # TODO
        # CHECK il resto del metodo
        self.POD.clear()
        self.POD.store_multiple_snapshots(self.all_snap)
        (zz, n) = self.POD.apply(self.S, self.post_processing_folder + "eigs", self.M1, self.tol)
        if self.N == 0:
            self.Z = zz
            self.N += n
    
        else:
            N = self.N
            Z = np.hstack((self.Z,zz))
            self.POD.clear()
            self.POD.store_multiple_snapshots(Z)
            (self.Z, self.N) = self.POD.apply(self.S, self.post_processing_folder + "eigs", N+self.M2, self.tol)
        np.save(self.basis_folder + "basis", self.Z)
    
        
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
    # CHECK il resto del metodo, ma non serve?
        delta_max = -1.0
        munew = None
        for mu in self.xi_train:
            self.setmu(mu)
            ParabolicCoerciveBase.online_solve(self,self.N,False)
            delta = self.get_delta()
            if delta > delta_max:
                delta_max = delta
                munew = mu
        print "absolute delta max = ", delta_max
        if os.path.isfile(self.post_processing_folder + "delta_max.npy") == True:
            d = np.load(self.post_processing_folder + "delta_max.npy")
            
            np.save(self.post_processing_folder + "delta_max", np.append(d, delta_max))
    
            m = np.load(self.post_processing_folder + "mu_greedy.npy")
            np.save(self.post_processing_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.post_processing_folder + "delta_max", delta_max)
            np.save(self.post_processing_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        
    ## Compute dual terms
    def compute_dual_terms(self):
    # CHECK il resto del metodo
        N = self.N
        RBu = Function(self.V)
        
        Qf = self.Qf
        Qa = self.Qa
        Qm = self.Qm
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



        self.CL = np.zeros((self.N,Qf,Qa))
        self.LL = np.zeros((self.N,self.N,self.Qa,self.Qa))
        self.CM = np.zeros((self.N,Qf,Qm))
        self.MM = np.zeros((self.N,self.N,self.Qm,self.Qm))
        self.LM = np.zeros((self.N,self.N,self.Qm,self.Qa))
        self.lnq = ()
        self.mnq = ()

        for n in range(self.N):
            RBu.vector()[:] = np.array(self.Z[:, n], dtype=np.float_)
            self.lnq += (self.compute_a_dual(RBu),)
            self.mnq += (self.compute_m_dual(RBu),)

        for n in range(self.N):
            la = Function(self.V)
            lap = Function(self.V)

            # CL
            for qf in range(0,Qf):
                for qa in range(0,Qa):
                    la.vector()[:] = np.array(self.lnq[n][:, qa], dtype=np.float_)
                    self.CL[n,qf,qa] = self.compute_scalar(self.Cf[qf],la,self.S)
            np.save(self.dual_folder + "CL", self.CL)
    
            # LL
            for qa in range(0,Qa):
                la.vector()[:] = np.array(self.lnq[n][:, qa], dtype=np.float_)
                for nn in range(0,N):
                    for qap in range(0,Qa):
                        lap.vector()[:] = np.array(self.lnq[nn][:, qap], dtype=np.float_)
                        self.LL[n,nn,qa,qap] = self.compute_scalar(la,lap,self.S)
                        if n != nn:
                            self.LL[nn,n,qa,qap] = self.LL[n,nn,qa,qap]
            np.save(self.dual_folder + "LL", self.LL)


            lm = Function(self.V)
            lmp = Function(self.V)


            # CM
            for qf in range(0,Qf):
                for qm in range(0,Qm):
                    lm.vector()[:] = np.array(self.mnq[n][:, qm], dtype=np.float_)
                    self.CM[n,qf,qm] = self.compute_scalar(self.Cf[qf],lm,self.S)
            np.save(self.dual_folder + "CM", self.CM)
    
            # MM
            for qm in range(0,Qm):
                lm.vector()[:] = np.array(self.mnq[n][:, qm], dtype=np.float_)
                for nn in range(0,N):
                    for qmp in range(0,Qm):
                        lmp.vector()[:] = np.array(self.mnq[nn][:, qmp], dtype=np.float_)
                        self.MM[n,nn,qm,qmp] = self.compute_scalar(lm,lmp,self.S)
                        if n != nn:
                            self.MM[nn,n,qm,qmp] = self.MM[n,nn,qm,qmp]
            np.save(self.dual_folder + "MM", self.MM)
    
            # LM
            for qm in range(0,Qm):
                lm.vector()[:] = np.array(self.mnq[n][:, qm], dtype=np.float_)
                for nn in range(0,N):
                    for qa in range(0,Qa):
                        la.vector()[:] = np.array(self.lnq[nn][:, qa], dtype=np.float_)
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
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Load reduced order data structures
    def load_reduced_matrices(self):
        # Read in data structures as in parents
        # (need to call them explicitly because this method was overridden in both parents)
        ParabolicCoerciveBase.load_reduced_matrices(self)
        EllipticCoerciveBase.load_reduced_matrices(self)
    
    #  @}
    ########################### end - I/O - end ###########################  
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    # Nothing to be added in this case
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ###########################
