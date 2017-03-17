# Copyright (C) 2015-2017 by the RBniCS authors
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
## @file elliptic_coercive_problem.py
#  @brief Base class for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from abc import ABCMeta, abstractmethod
import types
import hashlib
from RBniCS.problems.base.parametrized_problem import ParametrizedProblem
from RBniCS.backends import AffineExpansionStorage, copy, export, Function, import_
from RBniCS.utils.decorators import Extends, override, StoreMapFromProblemNameToProblem, StoreMapFromSolutionToProblem

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of elliptic coercive problems
@Extends(ParametrizedProblem) # needs to be first in order to override for last the methods.
@StoreMapFromProblemNameToProblem
@StoreMapFromSolutionToProblem
class ParametrizedDifferentialProblem(ParametrizedProblem):
    __metaclass__ = ABCMeta
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedProblem.__init__(self, type(self).__name__)
        
        # Input arguments
        self.V = V
        # Form names and order (to be filled in by child classes)
        self.terms = list()
        self.terms_order = dict()
        self.components = list()
        # Number of terms in the affine expansion
        self.Q = dict() # from string to integer
        # Matrices/vectors resulting from the truth discretization
        self.operator = dict() # from string to AffineExpansionStorage
        self.inner_product = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components), even though it will contain only one matrix
        self.dirichlet_bc = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
        self.dirichlet_bc_are_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
        # Solution
        self._solution = Function(self.V)
        self._solution_cache = dict() # of Functions
        self._output = 0
        # I/O
        self.folder["cache"] = self.folder_prefix + "/" + "cache"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Initialize data structures required for the offline phase
    def init(self):
        self._init_operators()
        self._init_dirichlet_bc()
        
    def _init_operators(self):
        # Get helper strings depending on the number of basis components
        n_components = len(self.components)
        assert n_components > 0
        if n_components > 1:
            inner_product_string = "inner_product_{c}"
        else:
            inner_product_string = "inner_product"
        # Assemble inner products
        if self.inner_product is None: # init was not called already
            inner_product = dict()
            for component in self.components:
                inner_product[component] = AffineExpansionStorage(self.assemble_operator(inner_product_string.format(c=component)))
            if n_components == 1:
                self.inner_product = inner_product.values()[0]
            else:
                self.inner_product = inner_product
        # Assemble operators
        for term in self.terms:
            self.operator[term] = AffineExpansionStorage(self.assemble_operator(term))
            self.Q[term] = len(self.operator[term])
            
    def _init_dirichlet_bc(self):
        # Get helper strings depending on the number of basis components
        n_components = len(self.components)
        assert n_components > 0
        if n_components > 1:
            dirichlet_bc_string = "dirichlet_bc_{c}"
        else:
            dirichlet_bc_string = "dirichlet_bc"
        # Assemble Dirichlet BCs
        # we do not assert for
        # (self.dirichlet_bc is None) == (self.dirichlet_bc_are_homogeneous is None)
        # because self.dirichlet_bc may still be None after initialization, if there
        # were no Dirichlet BCs at all and the problem had only one component
        if self.dirichlet_bc_are_homogeneous is None: # init was not called already
            dirichlet_bc = dict()
            dirichlet_bc_are_homogeneous = dict()
            for component in self.components:
                try:
                    operator_bc = AffineExpansionStorage(self.assemble_operator(dirichlet_bc_string.format(c=component)))
                except ValueError: # there were no Dirichlet BCs
                    dirichlet_bc[component] = None
                    dirichlet_bc_are_homogeneous[component] = False
                else:
                    dirichlet_bc[component] = operator_bc
                    try:
                        theta_bc = self.compute_theta(dirichlet_bc_string.format(c=component))
                    except ValueError: # there were no theta functions
                        # We provide in this case a shortcut for the case of homogeneous Dirichlet BCs,
                        # that do not require an additional lifting functions.
                        # The user needs to implement the dirichlet_bc case for assemble_operator, 
                        # but not the one in compute_theta (since theta would not matter, being multiplied by zero)
                        def generate_modified_compute_theta(component):
                            standard_compute_theta = self.compute_theta
                            def modified_compute_theta(self, term):
                                if term == dirichlet_bc_string.format(c=component):
                                    return (0,)*len(operator_bc)
                                else:
                                    return standard_compute_theta(term)
                            return modified_compute_theta
                        self.compute_theta = types.MethodType(generate_modified_compute_theta(component), self)
                        dirichlet_bc_are_homogeneous[component] = True
                    else:
                        dirichlet_bc_are_homogeneous[component] = False
            if n_components == 1:
                self.dirichlet_bc = dirichlet_bc.values()[0]
                self.dirichlet_bc_are_homogeneous = dirichlet_bc_are_homogeneous.values()[0]
            else:
                self.dirichlet_bc = dirichlet_bc
                self.dirichlet_bc_are_homogeneous = dirichlet_bc_are_homogeneous
    
    ## Perform a truth solve
    @abstractmethod
    def solve(self, **kwargs):
        (cache_key, cache_file) = self._cache_key_and_file_from_kwargs(**kwargs)
        if cache_key in self._solution_cache:
            assign(self._solution, self._solution_cache[cache_key])
        elif self.import_solution(self.folder["cache"], cache_file):
            self._solution_cache[cache_key] = copy(self._solution)
        else:
            assert not hasattr(self, "_is_solving")
            self._is_solving = True
            self._solve(**kwargs)
            delattr(self, "_is_solving")
            self._solution_cache[cache_key] = copy(self._solution)
        return self._solution
    
    ## Perform a truth solve. Internal method.
    @abstractmethod
    def _solve(self, **kwargs):
        raise NotImplementedError("The method _solve() is problem-specific and needs to be overridden.")
        
    ## Perform a truth evaluation of the output
    def compute_output(self):
        self._compute_output()
        return self._output
        
    ## Perform a truth evaluation of the output. Internal method.
    def _compute_output(self):
        self._output = NotImplemented
        
    def _cache_key_and_file_from_kwargs(self, **kwargs):
        for blacklist in ("components", "inner_product"):
            if blacklist in kwargs:
                del kwargs[blacklist]
        cache_key = (self.mu, tuple(sorted(kwargs.items())))
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        return (cache_key, cache_file)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Export solution to file
    def export_solution(self, folder, filename, solution=None, component=None):
        if solution is None:
            solution = self._solution
        assert component is None or isinstance(component, (str, list))
        if component is None and len(self.components) > 1:
            component = self.components
        if component is None:
            export(solution, folder, filename)
        elif isinstance(component, str):
            export(solution, folder, filename, component=component)
        elif isinstance(component, list):
            for c in component:
                assert isinstance(c, str)
                export(solution, folder, filename + "_" + c, component=c)
        else:
            raise AssertionError("Invalid component in export_solution()")
            
    ## Import solution from file
    def import_solution(self, folder, filename, solution=None):
        if solution is None:
            solution = self._solution
        return import_(solution, folder, filename)
        
    #  @}
    ########################### end - I/O - end ########################### 

    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of the problem.
    # Example of implementation:
    #   m1 = self.mu[0]
    #   m2 = self.mu[1]
    #   m3 = self.mu[2]
    #   if term == "a":
    #       theta_a0 = m1
    #       theta_a1 = m2
    #       theta_a2 = m1*m2+m3/7.0
    #       return (theta_a0, theta_a1, theta_a2)
    #   elif term == "f":
    #       theta_f0 = m1*m3
    #       return (theta_f0,)
    #   elif term == "dirichlet_bc":
    #       theta_bc0 = 1.
    #       return (theta_f0,)
    #   else:
    #       raise ValueError("Invalid term for compute_theta().")
    @abstractmethod
    def compute_theta(self, term):
        raise NotImplementedError("The method compute_theta() is problem-specific and needs to be overridden.")
        
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    # Example of implementation:
    #   if term == "a":
    #       a0 = inner(grad(u),grad(v))*dx
    #       return (a0,)
    #   elif term == "f":
    #       f0 = v*ds(1)
    #       return (f0,)
    #   elif term == "dirichlet_bc":
    #       bc0 = [(V, Constant(0.0), boundaries, 3)]
    #       return (bc0,)
    #   elif term == "inner_product":
    #       x0 = u*v*dx + inner(grad(u),grad(v))*dx
    #       return (x0,)
    #   else:
    #       raise ValueError("Invalid term for assemble_operator().")
    @abstractmethod
    def assemble_operator(self, term):
        raise NotImplementedError("The method assemble_operator() is problem-specific and needs to be overridden.")
        
    ## Return a lower bound for the coercivity constant
    # Example of implementation:
    #    return 1.0
    # Note that this method is not needed in POD-Galerkin reduced order models, and this is the reason
    # for which it is not marked as @abstractmethod
    def get_stability_factor(self):
        raise NotImplementedError("The method get_stability_factor() is problem-specific and needs to be overridden.")
    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

