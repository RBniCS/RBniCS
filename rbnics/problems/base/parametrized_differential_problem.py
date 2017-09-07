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

from abc import ABCMeta, abstractmethod
import types
import hashlib
from rbnics.problems.base.parametrized_problem import ParametrizedProblem
from rbnics.backends import AffineExpansionStorage, assign, copy, export, Function, import_, product, sum
from rbnics.utils.config import config
from rbnics.utils.decorators import Extends, override, StoreMapFromProblemNameToProblem, StoreMapFromProblemToTrainingStatus, StoreMapFromSolutionToProblem
from rbnics.utils.mpi import log, PROGRESS

# Base class containing the definition of elliptic coercive problems
@Extends(ParametrizedProblem) # needs to be first in order to override for last the methods.
@StoreMapFromProblemNameToProblem
@StoreMapFromProblemToTrainingStatus
@StoreMapFromSolutionToProblem
class ParametrizedDifferentialProblem(ParametrizedProblem, metaclass=ABCMeta):
    """
    Abstract class describing a parametrized differential problem.
    Inizialization of the solution space V, forms terms and their order, number of terms in the affine expansion Q, inner products and boundary conditions, truth solution.
    
    :param V: functional solution space.
    """
    
    @override
    def __init__(self, V, **kwargs):
    
        # Call to parent
        ParametrizedProblem.__init__(self, self.name())
        
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
        self._combined_inner_product = None
        self.projection_inner_product = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components), even though it will contain only one matrix
        self._combined_projection_inner_product = None
        self.dirichlet_bc = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components)
        self.dirichlet_bc_are_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
        self._combined_and_homogenized_dirichlet_bc = None
        # Solution
        self._solution = Function(self.V)
        self._solution_cache = dict() # of Functions
        self._output = 0
        self._output_cache = dict() # of floats
        self._output_cache__current_cache_key = None
        # I/O
        self.folder["cache"] = self.folder_prefix + "/" + "cache"
        self.cache_config = config.get("problems", "cache")
        
    def name(self):
        return type(self).__name__
    
    def init(self):
        """
        Calls _init_operators() and _init_dirichlet_bc(). 
        """
        self._init_operators()
        self._init_dirichlet_bc()
        
    def _init_operators(self):
        """
        Initialize operators required for the offline phase. Internal method.
        """
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
            assert self._combined_inner_product is None
            self._combined_inner_product = self._combine_all_inner_products()
        # Assemble inner product to be used for projection
        if self.projection_inner_product is None: # init was not called already
            projection_inner_product = dict()
            for component in self.components:
                try:
                    projection_inner_product[component] = AffineExpansionStorage(self.assemble_operator("projection_" + inner_product_string.format(c=component)))
                except ValueError: # no projection_inner_product specified, revert to inner_product
                    projection_inner_product[component] = AffineExpansionStorage(self.assemble_operator(inner_product_string.format(c=component)))
            if n_components == 1:
                self.projection_inner_product = projection_inner_product.values()[0]
            else:
                self.projection_inner_product = projection_inner_product
            assert self._combined_projection_inner_product is None
            self._combined_projection_inner_product = self._combine_all_projection_inner_products()
        # Assemble operators
        for term in self.terms:
            if term not in self.operator: # init was not called already
                self.operator[term] = AffineExpansionStorage(self.assemble_operator(term))
            if term not in self.Q: # init was not called already
                self.Q[term] = len(self.operator[term])
            
    def _combine_all_inner_products(self):
        if len(self.components) > 1:
            all_inner_products = list()
            for component in self.components:
                assert len(self.inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                all_inner_products.append(self.inner_product[component][0])
            all_inner_products = tuple(all_inner_products)
        else:
            assert len(self.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
            all_inner_products = (self.inner_product[0], )
        all_inner_products = AffineExpansionStorage(all_inner_products)
        all_inner_products_thetas = (1.,)*len(all_inner_products)
        return sum(product(all_inner_products_thetas, all_inner_products))
        
    def _combine_all_projection_inner_products(self):
        if len(self.components) > 1:
            all_projection_inner_products = list()
            for component in self.components:
                assert len(self.projection_inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                all_projection_inner_products.append(self.projection_inner_product[component][0])
            all_projection_inner_products = tuple(all_projection_inner_products)
        else:
            assert len(self.projection_inner_product) == 1 # the affine expansion storage contains only the inner product matrix
            all_projection_inner_products = (self.projection_inner_product[0], )
        all_projection_inner_products = AffineExpansionStorage(all_projection_inner_products)
        all_projection_inner_products_thetas = (1.,)*len(all_projection_inner_products)
        return sum(product(all_projection_inner_products_thetas, all_projection_inner_products))
        
    def _init_dirichlet_bc(self):
        """
        Initialize boundary conditions required for the offline phase. Internal method.
        """
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
                            def modified_compute_theta(self_, term):
                                if term == dirichlet_bc_string.format(c=component):
                                    return (0.,)*len(operator_bc)
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
            assert self._combined_and_homogenized_dirichlet_bc is None
            self._combined_and_homogenized_dirichlet_bc = self._combine_and_homogenize_all_dirichlet_bcs()
                
    def _combine_and_homogenize_all_dirichlet_bcs(self):
        if len(self.components) > 1:
            all_dirichlet_bcs = list()
            for component in self.components:
                if self.dirichlet_bc[component] is not None:
                    all_dirichlet_bcs.extend(self.dirichlet_bc[component])
            if len(all_dirichlet_bcs) > 0:
                all_dirichlet_bcs = tuple(all_dirichlet_bcs)
                all_dirichlet_bcs = AffineExpansionStorage(all_dirichlet_bcs)
            else:
                all_dirichlet_bcs = None
        else:
            all_dirichlet_bcs = self.dirichlet_bc
        if all_dirichlet_bcs is not None:
            all_dirichlet_bcs_thetas = (0.,)*len(all_dirichlet_bcs)
            return sum(product(all_dirichlet_bcs_thetas, all_dirichlet_bcs))
        else:
            return None
    
    def solve(self, **kwargs):
        """
        Perform a truth solve in case no precomputed solution is imported.
        """
        (cache_key, cache_file) = self._cache_key_and_file_from_kwargs(**kwargs)
        if "RAM" in self.cache_config and cache_key in self._solution_cache:
            log(PROGRESS, "Loading truth solution from cache")
            assign(self._solution, self._solution_cache[cache_key])
        elif "Disk" in self.cache_config and self.import_solution(self.folder["cache"], cache_file):
            log(PROGRESS, "Loading truth solution from file")
            if "RAM" in self.cache_config:
                self._solution_cache[cache_key] = copy(self._solution)
        else: # No precomputed solution available. Truth solve is performed.
            log(PROGRESS, "Solving truth problem")
            assert not hasattr(self, "_is_solving")
            self._is_solving = True
            assign(self._solution, Function(self.V))
            self._solve(**kwargs)
            delattr(self, "_is_solving")
            if "RAM" in self.cache_config:
                self._solution_cache[cache_key] = copy(self._solution)
            self.export_solution(self.folder["cache"], cache_file) # Note that we export to file regardless of config options, because they may change across different runs
        return self._solution
    
    class ProblemSolver(object, metaclass=ABCMeta):
        def __init__(self, problem):
            self.problem = problem
            
        def bc_eval(self):
            problem = self.problem
            if len(problem.components) > 1:
                all_dirichlet_bcs = list()
                all_dirichlet_bcs_thetas = list()
                for component in problem.components:
                    if problem.dirichlet_bc[component] is not None:
                        all_dirichlet_bcs.extend(problem.dirichlet_bc[component])
                        all_dirichlet_bcs_thetas.extend(problem.compute_theta("dirichlet_bc_" + component))
                if len(all_dirichlet_bcs) > 0:
                    all_dirichlet_bcs = tuple(all_dirichlet_bcs)
                    all_dirichlet_bcs = AffineExpansionStorage(all_dirichlet_bcs)
                    all_dirichlet_bcs_thetas = tuple(all_dirichlet_bcs_thetas)
                else:
                    all_dirichlet_bcs = None
                    all_dirichlet_bcs_thetas = None
            else:
                if problem.dirichlet_bc is not None:
                    all_dirichlet_bcs = problem.dirichlet_bc
                    all_dirichlet_bcs_thetas = problem.compute_theta("dirichlet_bc")
                else:
                    all_dirichlet_bcs = None
                    all_dirichlet_bcs_thetas = None
            assert (all_dirichlet_bcs is None) == (all_dirichlet_bcs_thetas is None)
            if all_dirichlet_bcs is not None:
                return sum(product(all_dirichlet_bcs_thetas, all_dirichlet_bcs))
            else:
                return None
        
        @abstractmethod
        def solve(self):
            pass
    
    ## Perform a truth solve
    @override
    def _solve(self, **kwargs):
        problem_solver = self.ProblemSolver(self)
        problem_solver.solve()
        
    def compute_output(self):
        """
        
        :return: output evaluation.
        """
        cache_key = self._output_cache__current_cache_key
        if "RAM" in self.cache_config and cache_key in self._output_cache:
            log(PROGRESS, "Loading truth output from cache")
            self._output = self._output_cache[cache_key]
        else: # No precomputed output available. Truth output is performed.
            log(PROGRESS, "Computing truth output")
            self._compute_output()
            if "RAM" in self.cache_config:
                self._output_cache[cache_key] = self._output
        return self._output
        
    def _compute_output(self):
        """
        Perform a truth evaluation of the output. Internal method.
        """
        self._output = NotImplemented
        
    def _cache_key_and_file_from_kwargs(self, **kwargs):
        """
        
        """
        for blacklist in ("components", "inner_product"):
            if blacklist in kwargs:
                del kwargs[blacklist]
        cache_key = (self.mu, tuple(sorted(kwargs.items())))
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        # Store current cache_key to be used when computing output
        self._output_cache__current_cache_key = cache_key
        # Return
        return (cache_key, cache_file)
    
    def export_solution(self, folder, filename, solution=None, component=None, suffix=None):
        """
        Export solution to file.
        """
        if solution is None:
            solution = self._solution
        assert component is None or isinstance(component, (str, list))
        if component is None and len(self.components) > 1:
            component = self.components
        if component is None:
            export(solution, folder, filename, suffix)
        elif isinstance(component, str):
            export(solution, folder, filename + "_" + component, suffix, component)
        elif isinstance(component, list):
            for c in component:
                assert isinstance(c, str)
                export(solution, folder, filename + "_" + c, suffix, c)
        else:
            raise AssertionError("Invalid component in export_solution()")
            
    def import_solution(self, folder, filename, solution=None, component=None, suffix=None):
        """
        Import solution from file.
        """
        if solution is None:
            solution = self._solution
        assert component is None or isinstance(component, (str, list))
        if component is None and len(self.components) > 1:
            component = self.components
        if component is None:
            return import_(solution, folder, filename, suffix)
        elif isinstance(component, str):
            return import_(solution, folder, filename + "_" + component, suffix, component)
        elif isinstance(component, list):
            for c in component:
                assert isinstance(c, str)
                if not import_(solution, folder, filename + "_" + c, suffix, c):
                    return False
            return True
        else:
            raise AssertionError("Invalid component in import_solution()")
    
    @abstractmethod
    def compute_theta(self, term):
        """Return theta multiplicative terms of the affine expansion of the problem.
        Example of implementation for Poisson problem:
           m1 = self.mu[0]
           m2 = self.mu[1]
           m3 = self.mu[2]
           if term == "a":
               theta_a0 = m1
               theta_a1 = m2
               theta_a2 = m1*m2+m3/7.0
               return (theta_a0, theta_a1, theta_a2)
           elif term == "f":
               theta_f0 = m1*m3
               return (theta_f0,)
           elif term == "dirichlet_bc":
               theta_bc0 = 1.
               return (theta_f0,)
           else:
               raise ValueError("Invalid term for compute_theta().")
        """
        raise NotImplementedError("The method compute_theta() is problem-specific and needs to be overridden.")
        
    
    @abstractmethod
    def assemble_operator(self, term):
        """ Return forms resulting from the discretization of the affine expansion of the problem operators.
         Example of implementation for Poisson problem:
           if term == "a":
               a0 = inner(grad(u),grad(v))*dx
               return (a0,)
           elif term == "f":
               f0 = v*ds(1)
               return (f0,)
           elif term == "dirichlet_bc":
               bc0 = [(V, Constant(0.0), boundaries, 3)]
               return (bc0,)
           elif term == "inner_product":
               x0 = u*v*dx + inner(grad(u),grad(v))*dx
               return (x0,)
           else:
               raise ValueError("Invalid term for assemble_operator().")
        """
        raise NotImplementedError("The method assemble_operator() is problem-specific and needs to be overridden.")
        
    def get_stability_factor(self):
        """Return a lower bound for the coercivity constant
         Example of implementation:
            return 1.0
         Note that this method is not needed in POD-Galerkin reduced order models, and this is the reason for which it is not marked as @abstractmethod
        """
        raise NotImplementedError("The method get_stability_factor() is problem-specific and needs to be overridden.")
            
