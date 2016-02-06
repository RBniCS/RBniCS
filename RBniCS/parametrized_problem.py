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
## @file parametrized_problem.py
#  @brief Implementation of a class containing an offline/online decomposition of parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import File, plot
import os # for path and makedir

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParametrizedProblem
#
# Base class containing an offline/online decomposition of parametrized problems
class ParametrizedProblem(object):
    """This is the base class, which is inherited by all other
    classes. It defines the base interface with variables and
    functions that the derived classes have to set and/or
    overwrite. The end user should not care about the implementation
    of this very class but he/she should derive one of the Elliptic or
    Parabolic class for solving an actual problem.

    The following functions are implemented:

    ## Set properties of the reduced order approximation
    - setNmax()
    - settol()
    - setmu_range()
    - setxi_train()
    - setxi_test()
    - generate_train_or_test_set()
    - setmu()
    
    ## Input/output methods
    - preprocess_solution_for_plot() # nothing to be done by default
    - move_mesh() # nothing to be done by default
    - reset_reference() # nothing to be done by default

    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self):
        # $$ ONLINE DATA STRUCTURES $$ #
        # 1. Online reduced space dimension
        self.N = 0
        # 2. Current parameters value
        self.mu = tuple()
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # 1. Maximum reduced order space dimension or tolerance to be used for the stopping criterion in the basis selection
        self.Nmax = 10
        self.tol = 1.e-15
        # 2. Parameter ranges and training set
        self.mu_range = list()
        self.xi_train = ParameterSpaceSubset()
        # 9. I/O
        self.xi_train_folder = "xi_train/"
        self.xi_test_folder = "xi_test/"
        
        # $$ ERROR ANALYSIS DATA STRUCTURES $$ #
        # 2. Test set
        self.xi_test = ParameterSpaceSubset()
    
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
        if enable_import:
            import_successful = self.xi_train.load(self.xi_train_folder, "xi_train") \
                and  (len(self.xi_train) == ntrain)
        if not import_successful:
            self.xi_train.generate(self.mu_range, ntrain, sampling)
            # Export 
            self.xi_train.save(self.xi_train_folder, "xi_train")
        
    ## ERROR ANALYSIS: set the elements in the test set \xi_test.
    # See the documentation of generate_train_or_test_set for more details
    def setxi_test(self, ntest, enable_import=False, sampling="random"):
        # Create I/O folder
        if not os.path.exists(self.xi_test_folder):
            os.makedirs(self.xi_test_folder)
        # Test if can import
        import_successful = False
        if enable_import:
            import_successful = self.xi_test.load(self.xi_test_folder, "xi_test") \
                and  (len(self.xi_test) == ntest)
        if not import_successful:
            self.xi_test.generate(self.mu_range, ntest, sampling)
            # Export 
            self.xi_test.save(self.xi_test_folder, "xi_test")
    
    ## OFFLINE/ONLINE: set the current value of the parameter
    def setmu(self, mu):
        assert (len(mu) == len(self.mu_range)), "mu and mu_range must have the same lenght"
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

