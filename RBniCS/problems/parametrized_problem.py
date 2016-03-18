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
#  @brief Implementation of a class containing basic definitions of parametrized problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import File, plot

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ParametrizedProblem
#
# Implementation of a class containing basic definitions of parametrized problems
class ParametrizedProblem(object):
    """This is the base class, which is inherited by all other
    classes. It defines the base interface with variables and
    functions that the derived classes have to set and/or
    overwrite. The end user should not care about the implementation
    of this very class but he/she should derive one of the Elliptic or
    Parabolic class for solving an actual problem.

    The following functions are implemented:

    ## Set properties of the reduced order approximation
    - set_Nmax()
    - settol()
    - set_mu_range()
    - set_xi_train()
    - set_xi_test()
    - generate_train_or_test_set()
    - set_mu()
    
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
        # 2. Current parameters value
        self.mu = tuple()
        # 2. Parameter ranges and training set
        self.mu_range = list()
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    ## OFFLINE: set the range of the parameters
    def set_mu_range(self, mu_range):
        self.mu_range = mu_range
    
    ## OFFLINE/ONLINE: set the current value of the parameter
    def set_mu(self, mu):
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
    def _export_vtk(self, solution, filename, **output_options):
        if not "with_mesh_motion" in output_options:
            output_options["with_mesh_motion"] = False
        if not "with_preprocessing" in output_options:
            output_options["with_preprocessing"] = False
        #
        file = File(filename + ".pvd", "compressed")
        if output_options["with_mesh_motion"]:
            self.move_mesh() # deform the mesh
        if output_options["with_preprocessing"]:
            preprocessed_solution = self.preprocess_solution_for_plot(solution)
            file << preprocessed_solution
        else:
            file << solution
        if output_options["with_mesh_motion"]:
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

