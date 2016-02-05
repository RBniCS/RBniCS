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
## @file parametrized_problem.py
#  @brief Implementation of a class containing an offline/online decomposition of parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import File, plot
import os # for path and makedir
import numpy as np
import itertools # for linspace sampling

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParametrizedProblem
#
# Base class containing an offline/online decomposition of parametrized problems
class ParametrizedProblem(object):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self):
        # $$ ONLINE DATA STRUCTURES $$ #
        # 1. Online reduced space dimension
        self.N = 0
        # 2. Current parameter
        self.mu = ()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 1. Maximum reduced order space dimension or tolerance to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        self.tol = 1.e-15
        # 2. Parameter ranges and training set
        self.mu_range = []
        self.xi_train = []
        # 9. I/O
        self.xi_train_folder = "xi_train/"
        self.xi_test_folder = "xi_test/"
        
        # $$ ERROR ANALYSIS DATA STRUCTURES $$ #
        # 2. Test set
        self.xi_test = []
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set maximum reduced space dimension (stopping criterion)
    def setNmax(self, nmax):
        self.Nmax = nmax
        
    ## OFFLINE: set tolerance of the offline phase (stopping criterion)
    def settol(self, tol):
        self.tol = tol
    
    ## OFFLINE: set the range of the parameters
    def setmu_range(self, mu_range):
        self.mu_range = mu_range
    
    ## OFFLINE: set the elements in the training set \xi_train.
    # See the documentation of generate_train_or_test_set for more details
    def setxi_train(self, ntrain, enable_import=False, sampling="random"):
        # Create I/O folder
        if not os.path.exists(self.xi_train_folder):
            os.makedirs(self.xi_train_folder)
        # Test if can import
        import_successful = False
        if enable_import and os.path.exists(self.xi_train_folder + "xi_train.npy"):
            xi_train = np.load(self.xi_train_folder + "xi_train.npy")
            import_successful = (len(np.asarray(xi_train)) == ntrain)
        if import_successful:
            self.xi_train = list()
            for i in range(len(np.asarray(xi_train))):
                self.xi_train.append(tuple(xi_train[i, :]))
        else:
            self.xi_train = self.generate_train_or_test_set(ntrain, sampling)
            # Export 
            np.save(self.xi_train_folder + "xi_train", self.xi_train)
        
    ## ERROR ANALYSIS: set the elements in the test set \xi_test.
    # See the documentation of generate_train_or_test_set for more details
    def setxi_test(self, ntest, enable_import=False, sampling="random"):
        # Create I/O folder
        if not os.path.exists(self.xi_test_folder):
            os.makedirs(self.xi_test_folder)
        # Test if can import
        import_successful = False
        if enable_import and os.path.exists(self.xi_test_folder + "xi_test.npy"):
            xi_test = np.load(self.xi_test_folder + "xi_test.npy")
            import_successful = (len(np.asarray(xi_test)) == ntest)
        if import_successful:
            self.xi_test = list()
            for i in range(len(np.asarray(xi_test))):
                self.xi_test.append(tuple(xi_test[i, :]))
        else:
            self.xi_test = self.generate_train_or_test_set(ntest, sampling)
            # Export 
            np.save(self.xi_test_folder + "xi_test", self.xi_test)
    
    ## Internal method for generation of training or test sets
    # If the last argument is equal to "random", n parameters are drawn from a random uniform distribution
    # Else, if the last argument is equal to "linspace", (approximately) n parameters are obtained from a cartesian grid
    def generate_train_or_test_set(self, n, sampling):
        if sampling == "random":
            ss = "[("
            for i in range(len(self.mu_range)):
                ss += "np.random.uniform(self.mu_range[" + str(i) + "][0], self.mu_range[" + str(i) + "][1])"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ") for _ in range(" + str(n) +")]"
            xi = eval(ss)
            return xi
        elif sampling == "linspace":
            n_P_root = int(np.ceil(n**(1./len(self.mu_range))))
            ss = "itertools.product("
            for i in range(len(self.mu_range)):
                ss += "np.linspace(self.mu_range[" + str(i) + "][0], self.mu_range[" + str(i) + "][1], num = " + str(n_P_root) + ").tolist()"
                if i < len(self.mu_range)-1:
                    ss += ", "
                else:
                    ss += ")"
            itertools_xi = eval(ss)
            xi = []
            for mu in itertools_xi:
                xi += [mu]
            return xi
        else:
            sys.exit("Invalid sampling mode.")

    ## OFFLINE/ONLINE: set the current value of the parameter
    def setmu(self, mu):
        self.mu = mu
    
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Interactive plot
    def _plot(self, solution, *args, **kwargs):
        self.move_mesh() # possibly deform the mesh
        preprocessed_solution = self.preprocess_solution_for_plot(solution)
        plot(preprocessed_solution, *args, **kwargs) # call FEniCS plot
        self.reset_reference() # undo mesh motion
        
    ## Export in VTK format
    def _export_vtk(self, solution, filename, output_options={}):
        if not "With mesh motion" in output_options:
            output_options["With mesh motion"] = False
        if not "With preprocessing" in output_options:
            output_options["With preprocessing"] = False
        #
        file = File(filename + ".pvd", "compressed")
        if output_options["With mesh motion"]:
            self.move_mesh() # deform the mesh
        if output_options["With preprocessing"]:
            preprocessed_solution = self.preprocess_solution_for_plot(solution)
            file << preprocessed_solution
        else:
            file << solution
        if output_options["With mesh motion"]:
            self.reset_reference() # undo mesh motion
            
    ## Preprocess the solution before plotting (e.g. to add a lifting)
    def preprocess_solution_for_plot(self, solution):
        return solution # nothing to be done by default
        
    ## Deform the mesh as a function of the geometrical parameters
    def move_mesh(self):
        pass # nothing to be done by default
    
    ## Restore the reference mesh
    def reset_reference(self):
        pass # nothing to be done by default
                
    #  @}
    ########################### end - I/O - end ########################### 

