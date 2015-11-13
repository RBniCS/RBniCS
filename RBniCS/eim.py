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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
import numpy as np
import os # for path and makedir
import shutil # for rm
import glpk # for LB computation
import sys # for sys.float_info.max
import random # to randomize selection in case of equal error bound
from gram_schmidt import *
from parametrized_problem import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EIM
#
# Empirical interpolation method for the interpolation of parametrized functions
class EIM(ParametrizedProblem):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, parametrized_problem):
        # Call the parent initialization
        ParametrizedProblem.__init__(self)
        # Note: parametrized_problem.V may be a VectorFunctionSpace, but here
        #       we are interested in the interpolation of a scalar function.
        #       Create therefore a new (scalar) FunctionSpace, which will be
        #       saved in self.V
        self.V = FunctionSpace(parametrized_problem.V.mesh(), "Lagrange", 1)
        # Store the dof to vertex map to locate maximum of functions
        self.dof_to_vertex_map = dof_to_vertex_map(self.V)
        # Store the parametrized problem object
        self.parametrized_problem = parametrized_problem
        # Store a string containing the parametrized function, to be assigned by the user
        # Please use x[0], x[1], x[2] to denote physical coordinates {note: brackets, zero based!}, and
        #            mu_1, mu_2, mu_3 to denote parameters           {note: underscores, one based!}
        self.parametrized_function = "invalid"
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Define additional storage for EIM
        self.interpolation_points = np.array([]) # vector of interpolation points selected by the greedy
        self.interpolation_points_dof = np.array([]) # vector of dofs corresponding to interpolation points selected by the greedy
        self.interpolation_matrix = () # interpolation matrix
        self.interpolation_coefficients = np.array([]) # online solution
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 6. Basis functions matrix
        self.Z = []
        # 6bis. Declare a new matrix to store the snapshots
        self.snapshot_matrix = np.array([])
        # 9. I/O
        self.snap_folder = "snapshots__eim/"
        self.basis_folder = "basis__eim/"
        self.dual_folder = "dual__eim/" # never used
        self.red_matrices_folder = "red_matr__eim/"
        self.pp_folder = "pp__eim/" # post processing
        #
        self.mu_index = 0
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set maximum reduced space dimension (stopping criterion).
    #           Overridden to resize the interpolation matrix
    def setNmax(self, nmax):
        self.Nmax = nmax
    
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve.
    def online_solve(self, N=None):
        self.load_red_matrices()
        if N is None:
            N = self.N
        
        if N == 0:
            return # nothing to be done
        
        # Evaluate the function at interpolation points
        rhs = np.zeros((N))
        for p in range(N):
            rhs[p] = self.evaluate_parametrized_function_at_mu_and_x(self.mu, self.interpolation_points[p])
        
        # Extract the interpolation matrix
        lhs = self.interpolation_matrix[:N,:N]
        
        # Solve the interpolation problem
        self.interpolation_coefficients = np.linalg.solve(lhs, rhs)
    
    ## Return an error bound for the current solution
    def get_delta(self):
        N = self.interpolation_coefficients.size(0)
        
        # Exact function evaluation at the next point
        f_next_point = self.evaluate_parametrized_function_at_mu_and_x(self.mu, self.interpolation_points[N])
        
        # EIM function evaluation at the next point
        eim_next_point = np.dot(self.interpolation_coefficients, self.interpolation_matrix[N, :])
        
        return abs(f_next_point - eim_next_point)
        
    ## Call online_solve and then convert the result of online solve from numpy to a tuple
    def compute_interpolated_theta(self):
        self.online_solve()
        return tuple(self.interpolation_coefficients)
        
    ## Evaluate the parametrized function f(x; mu)
    def evaluate_parametrized_function_at_mu_and_x(self, mu, x):
        expression = self.evaluate_parametrized_function_at_mu(mu)
        out = np.array([0.])
        expression.eval(out, x)
        return out[0]
    
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of EIM
    def offline(self):
        # Interpolate the parametrized function on the mesh grid for all parameters in xi_train
        print "=============================================================="
        print "=             EIM preprocessing phase begins                 ="
        print "=============================================================="
        print ""
        if os.path.exists(self.pp_folder):
            shutil.rmtree(self.pp_folder)
        folders = (self.snap_folder, self.basis_folder, self.dual_folder, self.red_matrices_folder, self.pp_folder)
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f)
                
        run = 0
        for mu in self.xi_train:
            print ":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::"
            
            print "evaluate parametrized function"
            self.setmu(self.xi_train[run])
            f = self.evaluate_parametrized_function_at_mu(self.mu)
            self.snap = interpolate(f, self.V)
            self.export_solution(self.snap, self.snap_folder + "truth_" + str(run))
            
            print "update snapshot matrix"
            self.update_snapshot_matrix()

            print ""
            run += 1
        
        print "=============================================================="
        print "=             EIM preprocessing phase ends                   ="
        print "=============================================================="
        print ""
        
        print "=============================================================="
        print "=             EIM offline phase begins                       ="
        print "=============================================================="
        print ""
        
        # Arbitrarily start from the first parameter in the training set
        self.setmu(self.xi_train[0])
        self.mu_index = 0
        # Create empty files
        np.save(self.red_matrices_folder + "interpolation_points", self.interpolation_points)
        np.save(self.red_matrices_folder + "interpolation_points_dof", self.interpolation_points_dof)
        np.save(self.red_matrices_folder + "interpolation_matrix", self.interpolation_matrix)
        np.save(self.basis_folder + "basis", self.Z)
        # Resize the interpolation matrix
        self.interpolation_matrix = np.matrix(np.zeros((self.Nmax + 1, self.Nmax + 1)))
        
        for run in range(self.Nmax + 1): # the + 1 is needed for the error bound computation
            print ":::::::::::::::::::::::::::::: EIM run = ", run, " ::::::::::::::::::::::::::::::"
            
            print "load parametrized function for mu = ", self.mu
            self.load_snapshot()
            
            print "solve eim"
            self.online_solve()
            
            print "compute maximum interpolation error"
            (maximum_point, maximum_point_dof) = self.compute_maximum_interpolation_error()
            if self.N == 0:
                self.interpolation_points = np.array([maximum_point])
                self.interpolation_points_dof = np.array([maximum_point_dof])
            else:
                self.interpolation_points = np.vstack((self.interpolation_points, [maximum_point]))
                self.interpolation_points_dof = np.vstack((self.interpolation_points_dof, [maximum_point_dof]))
            np.save(self.red_matrices_folder + "interpolation_points", self.interpolation_points)
            np.save(self.red_matrices_folder + "interpolation_points_dof", self.interpolation_points_dof)
            
            print "update basis matrix"
            self.update_basis_matrix()
            
            print "update interpolation matrix"
            self.update_interpolation_matrix()
            
            if self.N < self.Nmax + 1:
                print "find next mu"
                self.greedy()
            else:
                self.greedy()

            print ""
            
        print "=============================================================="
        print "=             EIM offline phase ends                         ="
        print "=============================================================="
        print ""
        
    ## Update the snapshot matrix
    def update_snapshot_matrix(self):
        if self.snapshot_matrix.size == 0: # for the first snapshot
            self.snapshot_matrix = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, np.array(self.snap.vector()).reshape(-1, 1))) # add new snapshots as column vectors
            
    ## The truth_solve method just returns the precomputed snapshot
    def load_snapshot(self):
        mu = self.mu
        mu_index = self.mu_index
        if mu != self.xi_train[mu_index]:
            # There is something wrong if we are here...
            sys.exit("Should never arrive here")
        self.snap.vector()[:] = np.array(self.snapshot_matrix[:, mu_index], dtype=np.float)
    
    # Compute the interpolation error and its maximum location
    def compute_maximum_interpolation_error(self):
        # self.snap now contains the exact function evaluation (loaded by truth solve)
        # Compute the error (difference with the eim approximation)
        for n in range(self.N):
            for d in range(self.snap.vector().size()):
                self.snap.vector()[d] = float(self.snap.vector()[d] - self.interpolation_coefficients[n]*self.Z[d, n])
        
        # Locate the vertex of the mesh where the error is maximum
        maximum_error = 0.0
        maximum_point = None
        maximum_point_dof = None
        for dof_index in range(self.V.dim()):
            vertex_index = self.dof_to_vertex_map[dof_index]
            err = self.snap.vector()[dof_index]
            if (abs(err) > abs(maximum_error) or (abs(err) == abs(maximum_error) and random.random() >= 0.5)):
                maximum_error = err
                maximum_point = self.V.mesh().coordinates()[vertex_index]
                maximum_point_dof = dof_index
        
        # Normalize the function in self.snap
        self.snap.vector()[:] /= maximum_error
               
        # Return
        return (maximum_point, maximum_point_dof)
        
    ## Update basis matrix
    def update_basis_matrix(self):
        if self.N == 0:
            self.Z = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
        else:
            self.Z = np.hstack((self.Z, np.array(self.snap.vector()).reshape(-1, 1))) # add new basis functions as column vectors
        np.save(self.basis_folder + "basis", self.Z)
        self.export_basis(self.snap, self.basis_folder + "basis_" + str(self.N))
        self.N += 1
        
    ## Assemble the reduced order affine expansion (matrix). Overridden 
    #  to assemble the interpolation matrix
    def update_interpolation_matrix(self):
        for j in range(self.N):
            self.interpolation_matrix[self.N - 1, j] = self.evaluate_basis_function_at_dof(j, self.interpolation_points_dof[self.N - 1])
        np.save(self.red_matrices_folder + "interpolation_matrix", self.interpolation_matrix)
            
    ## Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        err_max = -1.0
        munew = None
        munew_index = None
        for i in range(len(self.xi_train)):
            self.setmu(self.xi_train[i])
            self.mu_index = i
            
            # Load the exact function evaluation
            self.load_snapshot()
            
            # ... the EIM approximation ...
            self.online_solve()
            
            # ... and compute the difference
            for n in range(self.N):
                for d in range(self.snap.vector().size()):
                    self.snap.vector()[d] = float(self.snap.vector()[d] - self.interpolation_coefficients[n]*self.Z[d, n])
            
            # Compute the maximum error
            err = self.snap.vector().norm("linf")
            
            if (err > err_max):
                err_max = err
                munew = self.xi_train[i]
                munew_index = i
        print "absolute err max = ", err_max
        if os.path.isfile(self.pp_folder + "err_max.npy") == True:
            d = np.load(self.pp_folder + "err_max.npy")
            
            np.save(self.pp_folder + "err_max", np.append(d, err_max))
    
            m = np.load(self.pp_folder + "mu_greedy.npy")
            np.save(self.pp_folder + "mu_greedy", np.append(m, munew))
        else:
            np.save(self.pp_folder + "err_max", err_max)
            np.save(self.pp_folder + "mu_greedy", np.array(munew))

        self.setmu(munew)
        self.mu_index = munew_index
        
    ## Return the basis functions as tuples of functions
    def assemble_mu_independent_interpolated_function(self):
        output = ()
        for n in range(self.N):
            fun = Function(self.V)
            fun.vector()[:] = np.array(self.Z[:, n], dtype=np.float)
            output += (fun,)
        return output
        
    ## Evaluate the parametrized function f(.; mu)
    def evaluate_parametrized_function_at_mu(self, mu):
        expression_s = "Expression(\"" + self.parametrized_function + "\""
        for i in range(len(mu)):
            expression_s += ", mu_" + str(i+1) + "=" + str(mu[i])
        expression_s += ")"
        expression = eval(expression_s)
        return expression
        
    ## Evaluate the b-th basis function at the point corresponding to dof d
    def evaluate_basis_function_at_dof(self, b, d):
        return self.Z[d, b]
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{

    ## Export solution in VTK format
    def export_solution(self, solution, filename):
        file = File(filename + ".pvd", "compressed")
        file << solution
        
    ## Export basis in VTK format. 
    def export_basis(self, basis, filename):
        file = File(filename + ".pvd", "compressed")
        file << basis
        
    def load_red_matrices(self):
        if len(np.asarray(self.interpolation_points)) == 0: # avoid loading multiple times
            self.interpolation_points = np.load(self.red_matrices_folder + "interpolation_points.npy")
        if len(np.asarray(self.interpolation_points_dof)) == 0: # avoid loading multiple times
            self.interpolation_points_dof = np.load(self.red_matrices_folder + "interpolation_points_dof.npy")
        if len(np.asarray(self.interpolation_matrix)) == 0: # avoid loading multiple times
            self.interpolation_matrix = np.load(self.red_matrices_folder + "interpolation_matrix.npy")
        if len(np.asarray(self.Z)) == 0: # avoid loading multiple times
            self.Z = np.load(self.basis_folder + "basis.npy")
            if len(np.asarray(self.Z)) > 0: # it will still be empty the first time the greedy is executed
                self.N = self.Z.shape[1]
        
    #  @}
    ########################### end - I/O - end ########################### 
    
