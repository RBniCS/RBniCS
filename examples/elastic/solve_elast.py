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
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation

def compute_scalar(v1,v2,M):
    return np.dot(v1.vector(),M*v2.vector() )


def aff_assemble(A_vec, theta_a):
    A_ = A_vec[0]*theta_a[0]
    for i in range(1,len(A_vec)):
        A_ += A_vec[i]*theta_a[i]
    return A_

def compute_theta_a(mu,muref):
    mu1 = mu[0]
    mu2 = mu[1]
    mu3 = mu[2]
    mu4 = mu[3]
    mu5 = mu[4]
    mu6 = mu[5]
    mu7 = mu[6]
    mu8 = mu[7]
    mu9 = 1.
    theta_a0 = mu1
    theta_a1 = mu2
    theta_a2 = mu3
    theta_a3 = mu4
    theta_a4 = mu5
    theta_a5 = mu6
    theta_a6 = mu7
    theta_a7 = mu8
    theta_a8 = mu9
    theta_a = (theta_a0, theta_a1 ,theta_a2 ,theta_a3 ,theta_a4 ,theta_a5 ,theta_a6 ,theta_a7 ,theta_a8)
    #print theta_a
    return theta_a
    
def compute_theta_f(mu,muref):
    mu1 = mu[8]
    mu2 = mu[9]
    mu3 = mu[10]
    theta_f0 = mu1
    theta_f1 = mu2
    theta_f2 = mu3
    return (theta_f0, theta_f1, theta_f2)

def greedy(mumin, mumax, munp, CC, CL, LL):
    delta_max = 0.0
    count = 1
    for i0 in np.linspace(mumin[0], mumax[0], munp[0]):
        for i1 in np.linspace(mumin[1], mumax[1], munp[1]):
            for i2 in np.linspace(mumin[2], mumax[2], munp[2]):
                for i3 in np.linspace(mumin[3], mumax[3], munp[3]):
                    for i4 in np.linspace(mumin[4], mumax[4], munp[4]):
                        for i5 in np.linspace(mumin[5], mumax[5], munp[5]):
                            for i6 in np.linspace(mumin[6], mumax[6], munp[6]):
                                for i7 in np.linspace(mumin[7], mumax[7], munp[7]):
                                    for i8 in np.linspace(mumin[8], mumax[8], munp[8]):
                                        for i9 in np.linspace(mumin[9], mumax[9], munp[9]):
                                            for i10 in np.linspace(mumin[10], mumax[10], munp[10]):
                                                mu = (i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10)
                                                theta_a = compute_theta_a(mu,muref)
                                                theta_f = compute_theta_f(mu,muref)
                                                uN = rb_solve(red_A, red_F, theta_a, theta_f)
                                                lb = 0
                                                ub = 0
                                                delta = get_delta(theta_a, theta_f, CC, CL, LL, uN, lb, ub)
                                                if delta > delta_max:
                                                    delta_max = delta
                                                    munew = mu
    print "absolute delta_max = ", delta_max
    if os.path.isfile(pp_folder + "delta_max.npy") == True:
        d = np.load(pp_folder + "delta_max.npy")
        
        np.save(pp_folder + "delta_max", np.append(d, delta_max))

        m = np.load(pp_folder + "mu_greedy.npy")
        np.save(pp_folder + "mu_greedy", np.append(m, munew))
    else:
        print "non esiste"
        np.save(pp_folder + "delta_max", delta_max)
        np.save(pp_folder + "mu_greedy", np.array(munew))
    return munew

def greedy2(mumin, mumax, munp, CC, CL, LL):
    delta_max = 0.0
    MU = np.load("MU.npy")
    for mu in MU:
        theta_a = compute_theta_a(mu,muref)
        theta_f = compute_theta_f(mu,muref)
        uN = rb_solve(red_A, red_F, theta_a, theta_f)
        lb = 0
        ub = 0
        delta = get_delta(theta_a, theta_f, CC, CL, LL, uN, lb, ub)
        if delta > delta_max:
            delta_max = delta
            munew = mu
    print "absolute delta_max = ", delta_max
    if os.path.isfile(pp_folder + "delta_max.npy") == True:
        d = np.load(pp_folder + "delta_max.npy")
        
        np.save(pp_folder + "delta_max", np.append(d, delta_max))

        m = np.load(pp_folder + "mu_greedy.npy")
        np.save(pp_folder + "mu_greedy", np.append(m, munew))
    else:
        print "non esiste"
        np.save(pp_folder + "delta_max", delta_max)
        np.save(pp_folder + "mu_greedy", np.array(munew))
    return munew

def get_delta(theta_a, theta_f, CC, CL, LL, uN, lb, ub):
    eps2 = get_eps2(theta_a, theta_f, CC, CL, LL, uN)
    alpha = get_alpha_lb(theta_a, lb, ub)
    return np.sqrt(np.abs(eps2)/alpha)

def get_eps2 (theta_a, theta_f, CC, CL, LL, uN):
    Qf = len(theta_f)
    Qa = len(theta_a)
    eps2 = 0.0
    for qf in range(Qf):
        for qfp in range(Qf):
            eps2 += theta_f[qf]*theta_f[qfp]*CC[qf,qfp]
#    eps2 += theta_f[0]*theta_f[0]*CC
#    print "eps2 + CC = ", eps2
    if N == 1:
        for qf in range(Qf):
            for qa in range(Qa):
                eps2 += 2.0*theta_f[qf]*uN*theta_a[qa]*CL[0][qf,qa]

#        print "eps2 + CL = ", eps2

        for qa in range(Qa):
            for qap in range(Qa):
                eps2 += theta_a[qa]*uN*theta_a[qap]*LL[0,0,qa,qap]
        
#        print "eps2 + LL = ", eps2
#        for qa in range(Qa):
#            for qap in range(Qa):
#                eps2 += theta_a[qa]*theta_a[qap]*LL[qa,qap]
    else:
        n=0
        for un in uN:
            for qf in range(Qf):
                for qa in range(Qa):
                    eps2 += 2.0* theta_f[qf]*theta_a[qa]*un*CL[n][qf,qa]
            n += 1

#        print "eps2 + CL = ", eps2
        n = 0

        for un in uN:
            for qa in range(Qa):
                np = 0
                for unp in uN:
                    for qap in range(Qa):
                        eps2 += theta_a[qa]*un*theta_a[qap]*unp*LL[n,np,qa,qap]
                    np += 1
            n += 1
#        print "eps2 + LL = ", eps2
    #print "eps2 = ", eps2
    return eps2


def GS(basis):
    last = len(basis)-1
    b = basis[last].copy()
    for i in range(last):
        proj = np.dot(np.dot(b,L2*basis[i])/np.dot(basis[i],L2*basis[i]),basis[i])
        b = b - proj 
    basis[last] = b/np.sqrt(np.dot(b,L2*b))
    return basis

def build_rb_matrices(A_vec, Z):
    red_A = ()
    i = 0
    for A in A_vec:
        #A = down_cast(A)
        A = as_backend_type(A)
        if len(Z.shape) == 1:
            red_A += (np.dot(A.mat().getValues(range(dim),range(dim)).dot(Z.T),Z),)
        else:
            red = np.matrix(np.dot(Z,np.matrix(np.dot(A.mat().getValues(range(dim),range(dim)),Z.T))))
            red_A += (red,)
            i += 1
    #print red_A
    return red_A

def build_rb_vectors(F_vec, Z):
    red_F = ()
    i = 0
    for F in F_vec:
        #A = down_cast(A)
        F = as_backend_type(F)
        red_f = np.dot(Z, F.vec().getValues(range(dim)) )
        red_F += (red_f,)
    #print red_F
    return red_F

def compute_a_dual(A_vec, RBu):
    riez_a = ()
    riez = Function(V)
    i = 0
    for A in A_vec:
        solve (S,riez.vector(), A*RBu.vector()*(-1.0))
    #    plot(riez, interactive = True)
        riez_a += (riez,)
        if i != 0:
            l = np.vstack((l,riez.vector()))
        else:
            l = np.array(riez.vector())
            i = 1
    #for r in l:
    #    riez.vector()[:] = r
    #    plot(riez, interactive=True)
    return l
    return riez_a
def compute_f_dual(F_vec):
    riez_f0 = Function(V)
    riez_f1 = Function(V)
    riez_f2 = Function(V)
    solve (scalar == f0, riez_f0)
    solve (scalar == f1, riez_f1)
    solve (scalar == f2, riez_f2)
    riez = np.array(riez_f0.vector())
    riez = np.vstack((riez,np.array(riez_f1.vector())))
    riez = np.vstack((riez,np.array(riez_f2.vector())))
    riez = (riez_f0, riez_f1, riez_f2)
    return riez


def compute_dual_terms(A_vec, Z):
    global Cf
    global CC
    global CL
    global LL
    global lnq
    RBu = Function(V)
    riez = Function(V)
    riez_ = Function(V)
    if N == 1 :

        Cf = compute_f_dual(F_vec)
#        CC = compute_scalar(Cf,Cf,S)
        if Qf > 1:
            CC = np.zeros((Qf,Qf))
            for qf in range(0,Qf):
                for qfp in range(qf,Qf):
                    #CC[qf,qfp] = assemble((Cf[qf]*Cf[qfp])*dx)
                    #CC[qf,qfp] = assemble(inner(Cf[qf],Cf[qfp])*dx)
                    CC[qf,qfp] = compute_scalar(Cf[qf],Cf[qfp],S)
                    if qf != qfp:
                        CC[qfp,qf] = CC[qf,qfp]
        #print "CC:"
        #print CC
        np.save(dual_folder + "CC", CC)

        RBu.vector()[:] = Z
        
        lnq = (compute_a_dual(A_vec, RBu),)
        #lnq = compute_a_dual(A_vec, RBu)

        la = Function(V)
        lap = Function(V)
        Qa = len(theta_a)
        CL = np.zeros((Qf,Qa))
        # quando N >1 CL_big = (CL,)
        # CL_big += (CL,)
        for qf in range(0,Qf):
            for qa in range(0,Qa):
                la.vector()[:] = lnq[0][qa]
                #CL[qf,qa] = compute_scalar(la,Cf,S)
                CL[qf,qa] = compute_scalar(la,Cf[qf],S)
        CL = (CL,)
        #print "CL:"
        #print CL
        np.save(dual_folder + "CL", CL)
        
        LL = np.zeros((N,N,Qa,Qa))
        for qa in range(0,Qa):
            la.vector()[:] = lnq[0][qa]
            for qap in range(qa,Qa):
                lap.vector()[:] = lnq[0][qap]
                LL[0,0,qa,qap] = compute_scalar(la,lap,S)
                if qa != qap:
                    LL[0,0,qap,qa] = LL[0,0,qa,qap]
        #print "LL:"
        #print LL
        np.save(dual_folder + "LL", LL)
    else:
        RBu.vector()[:] = Z[N-1]
        lnq += (compute_a_dual(A_vec, RBu),)
        la = Function(V)
        lap = Function(V)
        Qa = len(theta_a)
        cl = np.zeros((Qf,Qa))
        for qf in range(0,Qf):
            for qa in range(0,Qa):
                la.vector()[:] = lnq[N-1][qa]
                cl[qf,qa] = compute_scalar(Cf[qf],la,S)
        CL += (cl,)
        #print "CL:"
        #print CL
        np.save(dual_folder + "CL", CL)

        LL.resize((N,N,Qa,Qa), refcheck=False)
        for n in range(N):
            for qa in range(0,Qa):
                la.vector()[:] = lnq[n][qa]
                for nn in range(N):
                    for qap in range(Qa):
                        lap.vector()[:] = lnq[nn][qap]
                        LL[n,nn,qa,qap] = compute_scalar(la,lap,S)
        #print "LL:"
        #print LL
        np.save(dual_folder + "LL", LL)


def rb_solve(red_A, red_F, theta_a, theta_f):
#    red_A = build_rb_matrices(A_vec, Z)
#    rb_F = build_rb_vectors(F_vec, Z)
    rb_A = aff_assemble(red_A, theta_a)
    rb_F = aff_assemble(red_F, theta_f)
    if isinstance(rb_A, float) == True:
        uN = rb_F/rb_A
    else:
        uN = np.linalg.solve(rb_A, rb_F)
    #print "uN"
    #print uN
    return uN

def perform_scm (A_vec, theta, L2, H1):
    A_vec_sym = ()
    for AA in A_vec:
        A_sym = (AA + np.transpose(AA))*0.5
        A_vec_sym += (A_sym,)
    P = compute_P(A_vec_sym, theta, L2, H1)
    lb = compute_LB(A_vec_sym, P)
    ub = compute_UB(A_vec_sym, P)
    return lb, ub


def compute_P (A_vec, theta, L2, H1):
    i = 0
    A = A_vec[0]*0.0
    for i in range(0,len(A_vec)):
        A += A_vec[i]*theta[i]

    #A = down_cast(A)
    A = as_backend_type(A)
    L2 = as_backend_type(L2)
    eigensolver = SLEPcEigenSolver(A, L2)
    eigensolver.parameters["spectrum"] = "smallest real"
    eigensolver.parameters["problem_type"] = "gen_hermitian"
#    eigensolver.parameters["solver"] = "lanczos"
    eigensolver.parameters["tolerance"] = 1e-10
    eigensolver.parameters["maximum_iterations"] = 50
    eigensolver.parameters["verbose"] = True
#    eigensolver.parameters["spectral_transform"] = "shift-and-invert"
#    eigensolver.parameters["spectral_shift"] = 1e-4
    info(eigensolver.parameters,verbose = True)
    ##########################################
    ### Compute all eigenvalues of A x = \lambda x
    print "Computing eigenvalues. This can take a minute."
    eigensolver.solve(40)
    ## Extract first eigenpair
    r, c, rx, cx = eigensolver.get_eigenpair(0)
    
    print "Smallest eigenvalue: ", r
    
    lambda_min = r
    
    P = np.abs(lambda_min)*L2 + H1
    P = L2 + H1
    P = down_cast(P)
    return P

    
def compute_LB (A_vec, P):
    """
    Returns the vector containing the y_q lower bound
    """
    Na = len(A_vec)
    lb = np.zeros(Na)
    i = 0
    for AA in A_vec:
        A = down_cast(AA)
        eigensolver = SLEPcEigenSolver(A, P)
        eigensolver.parameters["spectrum"] = "smallest real"
#        eigensolver.parameters["solver"] = "lanczos"
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["tolerance"] = 1e-10
        eigensolver.parameters["maximum_iterations"] = 50
        eigensolver.parameters["verbose"] = True
#        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
#        eigensolver.parameters["spectral_shift"] = 1e-15
        info(eigensolver.parameters,verbose = True)
        ##########################################
        ### Compute all eigenvalues of A x = \lambda x
        print "Computing eigenvalues. This can take a minute."
        eigensolver.solve(40)
        ## Extract first eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(0)
        
        print "Smallest eigenvalue: ", r
        lb[i] = r
        i+=1
    
    np.save(scm_folder + "lb", lb)
    return lb

def compute_UB (A_vec, P):
    Na = len(A_vec)
    ub = np.zeros(Na)
    i=0
    for AA in A_vec:
        A = down_cast(AA)
        eigensolver = SLEPcEigenSolver(A, P)
        eigensolver.parameters["spectrum"] = "largest real"
#        eigensolver.parameters["solver"] = "lanczos"
        eigensolver.parameters["problem_type"] = "gen_hermitian"
        eigensolver.parameters["tolerance"] = 1e-10
        eigensolver.parameters["maximum_iterations"] = 50
        eigensolver.parameters["verbose"] = True
#        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
    #    eigensolver.parameters["spectral_shift"] = 1e-4
        info(eigensolver.parameters,verbose = True)
        ##########################################
        ### Compute all eigenvalues of A x = \lambda x
        print "Computing eigenvalues. This can take a minute."
        eigensolver.solve(10)
        ## Extract first eigenpair
        r, c, rx, cx = eigensolver.get_eigenpair(0)
        
        print "Largest eigenvalue: ", r
        ub[i] = r
        i+=1

    np.save(scm_folder + "ub", ub)
    return ub

def get_alpha_lb(theta_a, lb, ub):
    #return max(theta_a)
    return 1.0/10.0
    alpha = 0.0
    i = 0
    for ta in theta_a:
        if ta >= 0.0:
            #alpha += ta*lb[i]
            alpha += ta*ub[i]
        else:
            alpha += ta*ub[i]
        i += 1
    return alpha

def original(x1,x2,x3,mu,muref, sol):
    mref = 2.0
    m = mu
    y_new = x2.copy()
    for i in range(len(y)):
        if y_new[i] > 3.0/5.0+DOLFIN_EPS:
            y_new[i] = 3.0/5.0 - 3.0/10.0*m + m/2.0*y_new[i]
        #    y_new[i] = (15*mref + 9)/(25*mref) - (3*m + 9/5)/(5*mref) + ((m + 3/5)/mref - 3/(5*mref))*y_new[i]
            
        #y_new[i] = (15*mref + 9)/(25*mref) - (3*m + 9/5)/(5*mref) + ((m + 3/5)/mref - 3/(5*mref))*y_new[i]
    
    return [x1, y_new, x3]


parameters.linear_algebra_backend = 'PETSc'

# Create mesh and define function space
mesh = Mesh("coarse.xml")
subd = MeshFunction("size_t", mesh, "coarse_physical_region.xml")
bound = MeshFunction("size_t", mesh, "coarse_facet_region.xml")
#mesh = Mesh("tblock.xml")
#subd = MeshFunction("size_t", mesh, "tblock_physical_region.xml")
#bound = MeshFunction("size_t", mesh, "tblock_facet_region.xml")


mmu = [(np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),np.random.uniform(0.1,10.0),   np.random.uniform(-1.0,1.0),np.random.uniform(-1.0,1.0),np.random.uniform(-1.0,1.0)) for _ in range(10000)]

np.save("mmu", mmu)
plot(mesh,interactive=True)
x = mesh.coordinates()[:,0].copy()
y = mesh.coordinates()[:,1].copy()


#plot(mesh, interactive=True, title='mesh')
#plot(subd, interactive=True, title='subd')
#plot(bound, interactive=True, title='bound')

V = VectorFunctionSpace(mesh, "Lagrange", 1)
dim = V.dim()
N = 0 # number of basis functions
Nmax = 20

# 
# Define new measures associated with the interior domains and
# exterior boundaries
dx = Measure("dx")[subd]
ds = Measure("ds")[bound]
# 
# mu1 = 0.3
# mu2 = 2.0
# mur = 2.0
# mu3 = 5.0
# 
# 
# 
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((1.0, 0.0))

E  = 1.0
nu = 0.3

mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(v.geometric_dimension())
    #return 2.0*mu*(grad(v)) + lmbda*((grad(v)))*Identity(v.geometric_dimension())

#a = inner(sigma(u), grad(v))*dx(1) +1e-15*inner(u,v)*dx
#L = inner(f, v)*dx

ascm = inner(sigma(u), grad(v))*dx
a0 = inner(sigma(u), grad(v))*dx(1) +1e-15*inner(u,v)*dx
a1 = inner(sigma(u), grad(v))*dx(2) +1e-15*inner(u,v)*dx
a2 = inner(sigma(u), grad(v))*dx(3) +1e-15*inner(u,v)*dx
a3 = inner(sigma(u), grad(v))*dx(4) +1e-15*inner(u,v)*dx
a4 = inner(sigma(u), grad(v))*dx(5) +1e-15*inner(u,v)*dx
a5 = inner(sigma(u), grad(v))*dx(6) +1e-15*inner(u,v)*dx
a6 = inner(sigma(u), grad(v))*dx(7) +1e-15*inner(u,v)*dx
a7 = inner(sigma(u), grad(v))*dx(8) +1e-15*inner(u,v)*dx
a8 = inner(sigma(u), grad(v))*dx(9) +1e-15*inner(u,v)*dx

l = Constant((1e-11, 1e-11))
f0 = inner(f,v)*ds(2) + inner(l,v)*dx
f1 = inner(f,v)*ds(3) + inner(l,v)*dx 
f2 = inner(f,v)*ds(4) + inner(l,v)*dx
out0 = f0 + f1 + f2

Ascm = assemble(ascm)
A0 = assemble(a0)
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)
A5 = assemble(a5)
A6 = assemble(a6)
A7 = assemble(a7)
A8 = assemble(a8)

F0 = assemble(f0)
F1 = assemble(f1)
F2 = assemble(f2)

Out0 = assemble(out0)

c = Constant((0.0, 0.0))
bc = DirichletBC(V, c, bound, 6)

bc.apply(A0)
bc.apply(A1)
bc.apply(A2)
bc.apply(A3)
bc.apply(A4)
bc.apply(A5)
bc.apply(A6)
bc.apply(A7)
bc.apply(A8)
bc.zero(A1)
bc.zero(A2)
bc.zero(A3)
bc.zero(A4)
bc.zero(A5)
bc.zero(A6)
bc.zero(A7)
bc.zero(A8)
#bc.apply(F0)
#bc.apply(F1)
#bc.apply(F2)


A_vec = (A0, A1, A2, A3, A4, A5, A6, A7, A8)
F_vec = (F0, F1, F2)
Qa = len(A_vec)
Qf = len(F_vec)
scalar = inner(u,v)*dx + inner(grad(u),grad(v))*dx
S = assemble(scalar)
l2 = inner(u,v)*dx
L2 = assemble(l2)
L2 = as_backend_type(L2)

#L2 = PETScMatrix()
#assemble(l2, tensor = L2)
h1 = inner(grad(u),grad(v))*dx
H1 = PETScMatrix()
assemble(h1, tensor = H1)

############################## PARAMETERS ##################################################
mu = (0.1, 0.1, 1., 1., 1., 1., 1., 1., 0.5, 0.5, 0.5)  # first mu
muref = (0.1, 0.1, 1., 1., 1., 1., 1., 1., 0.5, 0.5, 0.5) # reference
mumin = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1., -1., -1.) # min
mumax = (10, 10, 10, 10, 10, 10, 10, 10, 1., 1., 1.) # max
munp = (2, 2, 2., 2., 2., 2., 2., 2., 2, 2, 2) # number of points for generate the space of parameters
snap_folder = "snapshots/"
basis_folder = "basis/"
dual_folder = "dual/"
rb_matrices_folder = "rb_matr/"
scm_folder = "scm/"
pp_folder = "pp/" # post processing
folders = (snap_folder, basis_folder, dual_folder, rb_matrices_folder, scm_folder)
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)


###########################################################################################

snap = Function(V)
snapt = Function(V)
tr = Function(V)
rb = Function(V)
er = Function(V)


#if os.path.exists(pp_folder):
#    shutil.rmtree(pp_folder)
#os.makedirs(pp_folder)
#theta_a = compute_theta_a(mu,muref)
#theta_f = compute_theta_f(mu,muref)
#lb, ub = perform_scm(A_vec, theta_a, L2, H1)
lb, ub = perform_scm((Ascm,), (1.0,), L2, H1)
#Nmax = 60
#for run in range(Nmax):
#    print "############################## run = ", run, " ######################################"
#    
#    A = aff_assemble(A_vec, theta_a)
#    F = aff_assemble(F_vec, theta_f)
#    
#    print "truth solve for mu = ", mu
#    solve(A, snap.vector(), F)
#    sfile = File(snap_folder + "snap" + str(run) +".xml")
#    sfile << snap
##    plot(snap, title='assembled')
##    solve(a == L, snapt)
##    plot(snapt, title='truth')
##    interactive()
#   # print "truth output of interest compute average"
#    #av = assemble(mu[1]*snap*ds(1))
#    #print "average = ", av
#    #av = np.dot(snap.vector(),F0)
#    #av = np.dot(snap.vector(),F)
#    #print "computed av = ", av
#    
#    print "update base"
#    if N == 0:
#        Z = np.array(snap.vector())
#        Z /= np.sqrt(np.dot(Z,L2*Z))
#      #  Z /= np.sqrt(assemble(snap*snap*dx))
#    #    Z = GS(Z)
#
#    else:
#        Z = np.vstack((Z,snap.vector()))
#        Z = GS(Z)
#    #    print Z
#    #    print Z[0] - Z[1]
#    N += 1
#    
#    np.save(basis_folder + "basis", Z)
#    print "build_rb matrices"
#    red_A = build_rb_matrices(A_vec, Z)
#    red_F = build_rb_vectors(F_vec, Z)
#    np.save(rb_matrices_folder + "red_A", red_A)
#    np.save(rb_matrices_folder + "red_F", red_F)
#    print "solve-rb"
#    uN = rb_solve(red_A,red_F,theta_a,theta_f)
#    print uN
#    if N==1:
#        sol = Z*uN
#    else:
#        sol = Z[0]*uN[0]
#        i = 1
#        for un in uN[1:]:
#            sol += Z[i]*un
#            i+=1
#    #rb.vector()[:] = sol
#    #er.vector()[:] = np.array(snap.vector())-sol
#    #plot(snap, title='Truth')
#    #plot(rb, title='rb')
#    #plot(er, title='error')
#    #interactive()
#    rb_F = aff_assemble(red_F, theta_f)
#    #red_out0 = build_rb_vectors((Out0,),Z)
#    #red_out = np.dot(uN,red_out0)
#    #red_out = np.dot(uN,rb_F)
#    #print "reduced av = ", red_out
#    
#    print "compute dual terms"
#    compute_dual_terms(A_vec,Z)
#    print "error= ", get_eps2 (theta_a, theta_f, CC, CL, LL, uN)
#    
#    mu = greedy(mumin, mumax, munp, CC, CL, LL)
#    print "mu=", mu
#    theta_a = compute_theta_a(mu,muref)
#    theta_f = compute_theta_f(mu,muref)
#
#print "----------------------------- OFFLINE: ENDED -------------------------------------------------------"

#Z = np.load(basis_folder + "basis.npy")
#print len(Z)
#Z = Z[0:Z.shape[0]-1,:]
#print len(Z)
#np.save(basis_folder + "basis", Z)
#print "build_rb matrices"
#red_A = build_rb_matrices(A_vec, Z)
#red_F = build_rb_vectors(F_vec, Z)
#np.save(rb_matrices_folder + "red_A", red_A)
#np.save(rb_matrices_folder + "red_F", red_F)


print "----------------------------- ONLINE: BEGINS -------------------------------------------------------"
nn=100
error_hist=np.zeros(nn)
error_tr=np.zeros(nn)
red_A = np.load(rb_matrices_folder + "red_A.npy")
red_F = np.load(rb_matrices_folder + "red_F.npy")
Z = np.load(basis_folder + "basis.npy")
red_out0 = build_rb_vectors((Out0,),Z)
N = 50
Nmax=53
if N < Nmax:
        ra = (red_A[0][:N,:N],)
        for r in red_A[1:]:
            ra += (r[:N,:N],)
        red_A = ra
        rf = (red_F[0][:N],)
        for r in red_F[1:]:
            rf += (r[:N],)
        red_F = rf
        red_out0 = (red_out0[0][:N],)
for j in range(nn):
    mu1 = np.random.uniform(0.1,10)
    mu2 = np.random.uniform(0.1,10)
    mu3 = np.random.uniform(0.1,10)
    mu4 = np.random.uniform(0.1,10)
    mu5 = np.random.uniform(0.1,10)
    mu6 = np.random.uniform(0.1,10)
    mu7 = np.random.uniform(0.1,10)
    mu8 = np.random.uniform(0.1,10)
    mu9 = np.random.uniform(-1.0,1.0)
    mu10 = np.random.uniform(-1.0,1.0)
    mu11 = np.random.uniform(-1.0,1.0)
    mu = (mu1,mu2,mu3,mu4,mu5,mu6,mu7,mu8,mu9,mu10,mu11)
    theta_a = compute_theta_a(mu,muref)
    theta_f = compute_theta_f(mu,muref)
    A = aff_assemble(A_vec, theta_a)
    F = aff_assemble(F_vec, theta_f)
    
    print "truth solve for mu = ", mu
    solve(A, tr.vector(), F)
    tr_out = np.dot(tr.vector(), Out0)
    print "truth out = ", np.dot(tr.vector(), Out0)
    uN = rb_solve(red_A, red_F, theta_a, theta_f)
    CC = np.load(dual_folder + "CC.npy")
    CL = np.load(dual_folder + "CL.npy")
    LL = np.load(dual_folder + "LL.npy")
    
    lb = 0
    ub = 0
    
    print "absolute error bound = ", get_delta(theta_a, theta_f, CC, CL, LL, uN, lb, ub)
                    
    red_out = np.dot(uN,red_out0[0])
    print "rom out = ", red_out
    i=0
    bfile = File(pp_folder + "basis" + str(i) +".pvd")
    snap.vector()[:] = Z[i]
    bfile << snap
    sol = Z[0]*uN[0]
    i = 1
    for un in uN[1:]:
        bfile = File(pp_folder + "basis" + str(i) +".pvd")
        snap.vector()[:] = Z[i]
        bfile << snap
        sol += Z[i]*un
        i+=1
    rb.vector()[:] = sol
    er.vector()[:] = np.array(tr.vector())-sol
    print "computed error = ", np.sqrt(assemble(inner(er,er)*dx))
    #plot(tr, title='Truth')
    #plot(rb, title='rb')
    #plot(er, title='error')
    rfile = File(pp_folder + "rb" + str(j) +".pvd")
    tfile = File(pp_folder + "truth" + str(j) +".pvd")
    efile = File(pp_folder + "error" + str(j) +".pvd")
    rfile << rb
    tfile << tr
    efile << er
    interactive()
    error_hist[j] = get_delta(theta_a, theta_f, CC, CL, LL, uN, lb, ub)**2
    error_tr[j] = np.abs(red_out-tr_out)

#print "error average=", np.mean(error_hist)
#print "error min=", np.min(error_hist)
#print "error max=", np.max(error_hist)


eta = error_hist/error_tr
print "delta_ave = ", np.mean(error_hist)
print "eta_max = ", np.max(eta)
print "eta_ave = ", np.mean(eta)
