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
import os # for path and makedir
import shutil # for rm
import glpk # for LB computation
import sys # for sys.float_info.max
from elliptic_coercive_rb_base import *

#~~~~~~~~~~~~~~~~~~~~~~~~~     EIM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EIM
#
# Empirical interpolation method for the interpolation of parametrized functions
class EIM(EllipticCoerciveRBBase):

    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{
    
    ## Default initialization of members
    def __init__(self, parametrized_problem):
        # Call the parent initialization
        # Note: parametrized_problem.V may be a VectorFunctionSpace, but here
        #       we are interested in the interpolation of a scalar function.
        #       Create therefore a new (scalar) FunctionSpace, which will be
        #       saved in self.V
        scalar_V = FunctionSpace(V.___mesh, V.___family, V.___degree)
        EllipticCoerciveRBBase.__init__(self, scalar_V)
        # Store the parametrized problem object
        self.parametrized_problem = parametrized_problem
        # Store a string containing the parametrized function, to be assigned by the user
        # Please use x[0], x[1], x[2] to denote physical coordinates {note: brackets, zero based!}, and
        #            mu_1, mu_2, mu_3 to denote parameters           {note: underscores, one based!}
        self.parametrized_function = "invalid"
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Define additional storage for EIM
        self.interpolation_points = [] # vector of interpolation points selected by the greedy
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 6bis. Declare a new matrix to store the snapshots
        self.snapshot_matrix = np.array([])
        # 9. I/O
        self.snap_folder = "snapshots__eim/"
        self.basis_folder = "basis__eim/"
        self.dual_folder = "dual__eim/" # never used
        self.red_matrices_folder = "red_matr__eim/"
        self.pp_folder = "pp__eim/" # post processing
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    def online_solve(self, mu, N=None, with_plot=True):
        # TODO implementare, aggiornare commento, e togliere i parametri opzionali inutili
        # Forse ne devi fare due versioni come in libmesh?
    
    ## Return an error bound for the current solution
    def get_delta(self):
        # TODO: vedi compute_best_fit_error di libMesh.. Pero non e' veramente online, perche' usa
        # anche la soluzione truth.. Guarda meglio...
        
    ## TODO
    def compute_interpolated_theta(self):
        # TODO
        
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
        print "=        EIM preprocessing phase begins                      ="
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
            print "§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§ ",self.name," run = ", run, " §§§§§§§§§§§§§§§§§§§§§§§§§§§§§§"
            
            print "evaluate parametrized function"
            self.setmu(self.xi_train[run])
            f = evaluate_parametrized_function_at_mu(self.xi_train[run])
            self.snap = Interpolate(f, self.V)
            
            print "update snapshot matrix"
            self.update_snapshot_matrix()

            print ""
            run += 1
        
        print "=============================================================="
        print "=        EIM preprocessing phase ends                        ="
        print "=============================================================="
        print ""
        
        # Afterwards, call the standard (parent) offline phase. Note however that we
        # redefine some methods employed internally by the greedy algorithm
        EllipticCoerciveRBBase.offline(self)
        
    ## Update the snapshot matrix
    def update_snapshot_matrix(self):
        if self.snapshot_matrix.size == 0: # for the first snapshot
            self.snapshot_matrix = np.array(self.snap.vector()).reshape(-1, 1) # as column vector
        else:
            self.snapshot_matrix = np.hstack((self.snapshot_matrix, self.snap.vector())) # add new snapshots as column vectors
            
    ## Override the truth_solve method: it just returns the precomputed snapshot
    def truth_solve(self):
        mu = self.mu
        for i in range(len(self.xi_train)):
            if self.xi_train[i] == mu:
                self.snap.vector()[:] = self.snapshot_matrix[:, i]
                print "(precomputed)"
                return
        # Make sure to handle also the case where mu was not found: for instance,
        # this may happen if we use truth_solve in the error computation
        print "(computing ...)"
        f = evaluate_parametrized_function_at_mu(mu)
        self.snap = Interpolate(f, self.V)
        
    ## Assemble the reduced order affine expansion (matrix). Overridden 
    #  to assemble the interpolation matrix
    def build_red_matrices(self):
        red_A.resize((self.N,self.N)) # TODO check se esiste
        for j in range(self.N):
            red_A[self.N - 1, j] = evaluate_parametrized_function_at_mu_and_x(mu, self.interpolation_points[j])
    
    ## Assemble the reduced order affine expansion (rhs)
    def build_red_vectors(self):
        # TODO boh, non serve? Evitare di ereditare da RB?
        # Se evitiamo, allora togliere il system.name
        
    ## Compute dual terms
    def compute_dual_terms(self):
        # TODO questi termini duali non servono... Evitare di ereditare da RB?
        
    ## TODO
    def assemble_mu_independent_interpolated_function(self):
        # TODO
        
    ## Evaluate the parametrized function f(.; mu)
    def evaluate_parametrized_function_at_mu(self, mu):
        expression_s = "Expression(\"" + parametrized_function + "\""
        for i in range(len(mu)):
            expression_s += ", mu_" + str(i+1) + "=" + str(mu[i])
        expression_s += ")"
        expression = eval(expression_s)
        return expression
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
