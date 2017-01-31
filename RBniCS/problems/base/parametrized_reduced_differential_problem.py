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
## @file elliptic_coercive_reduced_problem.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from __future__ import print_function
from abc import ABCMeta, abstractmethod
import types
from math import sqrt
from numpy import isclose
from RBniCS.problems.base.parametrized_problem import ParametrizedProblem
from RBniCS.backends import BasisFunctionsMatrix, transpose
from RBniCS.backends.online import OnlineAffineExpansionStorage, OnlineFunction
from RBniCS.sampling import ParameterSpaceSubset
from RBniCS.utils.decorators import Extends, override, StoreMapFromProblemToReducedProblem, sync_setters
from RBniCS.utils.mpi import print

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveReducedOrderModelBase
#
# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(ParametrizedProblem) # needs to be first in order to override for last the methods.
@StoreMapFromProblemToReducedProblem
class ParametrizedReducedDifferentialProblem(ParametrizedProblem):
    __metaclass__ = ABCMeta
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        ParametrizedProblem.__init__(self, type(truth_problem).__name__)
        
        # $$ ONLINE DATA STRUCTURES $$ #
        # Online reduced space dimension
        self.N = None # integer (for problems with one component) or dict of integers (for problem with several components)
        self.N_bc = None # integer (for problems with one component) or dict of integers (for problem with several components)
        self.dirichlet_bc = None # bool (for problems with one component) or dict of bools (for problem with several components)
        self.dirichlet_bc_are_homogeneous = None # bool (for problems with one component) or dict of bools (for problem with several components)
        # Form names and order
        self.terms = truth_problem.terms
        self.terms_order = truth_problem.terms_order
        self.components = truth_problem.components
        # Number of terms in the affine expansion
        self.Q = dict() # from string to integer
        # Reduced order operators
        self.operator = dict() # from string to OnlineAffineExpansionStorage
        self.inner_product = None # AffineExpansionStorage (for problems with one component) or dict of AffineExpansionStorage (for problem with several components), even though it will contain only one matrix
        # Solution
        self._solution = OnlineFunction()
        self._output = 0
        self._compute_error__previous_mu = None
        self._compute_error_output__previous_mu = None
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.truth_problem = truth_problem
        # Basis functions matrix
        self.Z = BasisFunctionsMatrix(truth_problem.V)
        # I/O
        self.folder["basis"] = self.folder_prefix + "/" + "basis"
        self.folder["reduced_operators"] = self.folder_prefix + "/" + "reduced_operators"
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Initialize data structures required for the online phase
    def init(self, current_stage="online"):
        self._init_operators(current_stage)
        self._init_basis_functions(current_stage)
            
    def _init_operators(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        if current_stage == "online":
            # Inner products
            n_components = len(self.components)
            if n_components > 1:
                inner_product_string = "inner_product_{c}"
                self.inner_product = dict()
                for component in self.components:
                    self.inner_product[component] = self.assemble_operator(inner_product_string.format(c=component), "online")
            else:
                self.inner_product = self.assemble_operator("inner_product", "online")
            # Terms
            for term in self.terms:
                self.operator[term] = self.assemble_operator(term, "online")
                self.Q[term] = len(self.operator[term])
        elif current_stage == "offline":
            # Inner products
            n_components = len(self.components)
            if n_components > 1:
                self.inner_product = dict()
                for component in self.components:
                    self.inner_product[component] = OnlineAffineExpansionStorage(1)
            else:
                self.inner_product = OnlineAffineExpansionStorage(1)
            # Terms
            for term in self.terms:
                self.Q[term] = self.truth_problem.Q[term]
                self.operator[term] = OnlineAffineExpansionStorage(self.Q[term])
        else:
            raise AssertionError("Invalid stage in _init_operators().")
        
    def _init_basis_functions(self, current_stage="online"):
        assert current_stage in ("online", "offline")
        # Initialize basis functions mappings
        self.Z.init(self.components)
        # Get number of components
        n_components = len(self.components)
        # Get helper strings depending on the number of basis components
        if n_components > 1:
            dirichlet_bc_string = "dirichlet_bc_{c}"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]
            def get_Z(component):
                return self.Z[component]
        else:
            dirichlet_bc_string = "dirichlet_bc"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous
            def get_Z(component):
                return self.Z
        # Detect how many theta terms are related to boundary conditions
        assert (self.dirichlet_bc is None) == (self.dirichlet_bc_are_homogeneous is None)
        if self.dirichlet_bc is None: # init was not called already
            dirichlet_bc = dict()
            for component in self.components:
                try:
                    theta_bc = self.compute_theta(dirichlet_bc_string.format(c=component))
                except ValueError: # there were no Dirichlet BCs to be imposed by lifting
                    dirichlet_bc[component] = False
                else:
                    dirichlet_bc[component] = True
            if n_components == 1:
                self.dirichlet_bc = dirichlet_bc.values()[0]
            else:
                self.dirichlet_bc = dirichlet_bc
            self.dirichlet_bc_are_homogeneous = self.truth_problem.dirichlet_bc_are_homogeneous
        # Load basis functions
        if current_stage == "online":
            Z_loaded = self.Z.load(self.folder["basis"], "basis")
            # To properly initialize N and N_bc, detect how many theta terms
            # are related to boundary conditions
            if Z_loaded:
                N = dict()
                N_bc = dict()
                for component in self.components:
                    if has_non_homogeneous_dirichlet_bc(component):
                        theta_bc = self.compute_theta(dirichlet_bc_string.format(c=component))
                        N[component] = len(get_Z(component)) - len(theta_bc)
                        N_bc[component] = len(theta_bc)
                    else:
                        N[component] = len(get_Z(component))
                        N_bc[component] = 0
                assert len(N) == len(N_bc)
                assert len(N) > 0
                if len(N) == 1:
                    self.N = N.values()[0]
                    self.N_bc = N_bc.values()[0]
                else:
                    self.N = N
                    self.N_bc = N_bc
        elif current_stage == "offline":
            # Store the lifting functions in self.Z
            for component in self.components:
                self.assemble_operator(dirichlet_bc_string.format(c=component), "offline") # no return value from assemble_operator in this case
            # Save basis functions matrix, that contains up to now only lifting functions
            self.Z.save(self.folder["basis"], "basis")
            # Properly fill in self.N_bc
            total_N_bc = 0
            if n_components == 1:
                self.N = 0
                self.N_bc = len(self.Z)
                total_N_bc = self.N_bc
            else:
                N = dict()
                N_bc = dict()
                for component in self.components:
                    N[component] = 0
                    N_bc[component] = len(self.Z[component])
                self.N = N
                self.N_bc = N_bc
                total_N_bc = sum(N_bc.values())
            # Note that, however, self.N is not increased, so it will actually contain the number
            # of basis functions without the lifting ones.
            if total_N_bc > 0:
                # Finally, since the solution for the current value of mu is already in the basis (at least for linear problems),
                # we arbitrarily generate a new value of mu to the minimum of the range
                new_mu = tuple([r[0] for r in self.mu_range])
                assert self.mu != new_mu
                self.set_mu(new_mu)
        else:
            raise AssertionError("Invalid stage in _init_basis_functions().")
            
    # Perform an online solve. self.N will be used as matrix dimension if the default value is provided for N.
    @override
    def solve(self, N=None, **kwargs):
        N, kwargs = self._online_size_from_kwargs(N, **kwargs)
        return self._solve(N, **kwargs)
        
    # Perform an online solve. Internal method
    @abstractmethod
    def _solve(self, N, **kwargs):
        raise NotImplementedError("The method _solve() is problem-specific and needs to be overridden.")
    
    # Perform an online evaluation of the output
    def output(self):
        self._output = NotImplemented
        return self._output
        
    def _online_size_from_kwargs(self, N, **kwargs):
        class OnlineSizeDict(dict):
            __slots__ = ()
            
            def __init__(self_, *args_, **kwargs_):
                super(OnlineSizeDict, self_).__init__(*args_, **kwargs_)
                
            def __getitem__(self_, k):
                return super(OnlineSizeDict, self_).__getitem__(k)
                
            def __setitem__(self_, k, v):
                return super(OnlineSizeDict, self_).__setitem__(k, v)
                
            def __delitem__(self_, k):
                return super(OnlineSizeDict, self_).__delitem__(k)
                
            def get(self_, k, default=None):
                return super(OnlineSizeDict, self_).get(k, default)
                
            def setdefault(self_, k, default=None):
                return super(OnlineSizeDict, self_).setdefault(k, default)
                
            def pop(self_, k):
                return super(OnlineSizeDict, self_).pop(k)
                
            def update(self_, **kwargs_):
                super(OnlineSizeDict, self_).update(**kwargs_)
                
            def __contains__(self_, k):
                return super(OnlineSizeDict, self_).__contains__(k)
                
            # Override N += N_bc so that it is possible to increment online size due to boundary conditions
            def __iadd__(self_, other_):
                for component in self.components:
                    self_[component] += other_[component]
                return self_
        
        if len(self.components) > 1:
            if N is None:
                all_components_in_kwargs = self.components[0] in kwargs
                for component in self.components:
                    if all_components_in_kwargs:
                        assert component in kwargs, "You need to specify the online size of all components in kwargs" 
                    else:
                        assert component not in kwargs, "You need to specify the online size of all components in kwargs"
                if all_components_in_kwargs:
                    N = OnlineSizeDict()
                    for component in self.components:
                        N[component] = kwargs[component]
                        del kwargs[component]
                else:
                    N = OnlineSizeDict(self.N) # copy the default dict
            else:
                assert isinstance(N, int)
                N_int = N
                N = OnlineSizeDict()
                for component in self.components:
                    N[component] = N_int
                    assert component not in kwargs, "You cannot provide both an int and kwargs for components"
        else:
            if N is None:
                assert len(self.components) == 1
                component_0 = self.components[0]
                if component_0 in kwargs:
                    N = kwargs[component_0]
                else:
                    N = self.N
            else:
                assert isinstance(N, int)
                
        return N, kwargs
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 

    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Assemble the reduced order affine expansion.
    def build_reduced_operators(self):
        # Terms
        for term in self.terms:
            self.operator[term] = self.assemble_operator(term, "offline")
        # Inner products
        n_components = len(self.components)
        if n_components > 1:
            inner_product_string = "inner_product_{c}"
            for component in self.components:
                self.inner_product[component] = self.assemble_operator(inner_product_string.format(c=component), "offline")
        else:
            self.inner_product = self.assemble_operator("inner_product", "offline")
        
    ## Postprocess a snapshot before adding it to the basis/snapshot matrix, for instance removing
    # non-homogeneous Dirichlet boundary conditions
    def postprocess_snapshot(self, snapshot, snapshot_index):
        n_components = len(self.components)
        # Get helper strings and functions depending on the number of basis components
        if n_components > 1:
            dirichlet_bc_string = "dirichlet_bc_{c}"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]
            def assert_lengths(component):
                assert self.N_bc[component] == len(theta_bc)
        else:
            dirichlet_bc_string = "dirichlet_bc"
            def has_non_homogeneous_dirichlet_bc(component):
                return self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous
            def assert_lengths(component):
                assert self.N_bc == len(theta_bc)
        # Carry out postprocessing
        for component in self.components:
            if has_non_homogeneous_dirichlet_bc(component):
                theta_bc = self.compute_theta(dirichlet_bc_string.format(c=component))
                assert_lengths(component)
                return snapshot - self.Z[:self.N_bc]*theta_bc
            else:
                return snapshot
        
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_error(self, N=None, **kwargs):
        if self._compute_error__previous_mu != self.mu:
            self.truth_problem.solve(**kwargs)
            self.truth_problem.output()
            # Do not carry out truth solves anymore for the same parameter
            self._compute_error__previous_mu = self.mu
        # Compute the error on the solution
        self.solve(N, **kwargs)
        return self._compute_error(**kwargs)
        
    # Internal method for error computation
    def _compute_error(self, **kwargs):
        (components, inner_product) = self._preprocess_compute_error_and_relative_error_kwargs(**kwargs)
        # Storage
        error = dict()
        # Compute the error on the solution
        if len(components) > 0:
            N = self._solution.N
            reduced_solution = self.Z[:N]*self._solution
            truth_solution = self.truth_problem._solution
            error_function = truth_solution - reduced_solution
            for component in components:
                error_norm_squared_component = transpose(error_function)*inner_product[component]*error_function
                assert error_norm_squared_component >= 0. or isclose(error_norm_squared_component, 0.)
                error[component] = sqrt(error_norm_squared_component)
        # Simplify trivial case
        if len(components) == 1:
            error = error[components[0]]
        #
        return error
        
    # Compute the relative error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_relative_error(self, N=None, **kwargs):
        absolute_error = self.compute_error(N, **kwargs)
        return self._compute_relative_error(absolute_error, **kwargs)
        
    # Internal method for relative error computation
    def _compute_relative_error(self, absolute_error, **kwargs):
        (components, inner_product) = self._preprocess_compute_error_and_relative_error_kwargs(**kwargs)
        # Handle trivial case from compute_error
        if len(components) == 1:
            absolute_error_ = dict()
            absolute_error_[components[0]] = absolute_error
            absolute_error = absolute_error_
        # Storage
        relative_error = dict()
        # Compute the relative error on the solution
        if len(components) > 0:
            truth_solution = self.truth_problem._solution
            for component in components:
                exact_norm_squared_component = transpose(truth_solution)*inner_product[component]*truth_solution
                assert exact_norm_squared_component >= 0. or isclose(exact_norm_squared_component, 0.)
                relative_error[component] = absolute_error[component]/sqrt(exact_norm_squared_component)
        # Simplify trivial case
        if len(components) == 1:
            relative_error = relative_error[components[0]]
        #
        return relative_error
                
    def _preprocess_compute_error_and_relative_error_kwargs(self, **kwargs):
        # Set default components, if needed
        if "components" not in kwargs:
            kwargs["components"] = self.components
        # Set inner product for components, if needed
        if "inner_product" not in kwargs:
            inner_product = dict()
            for component in kwargs["components"]:
                assert len(self.truth_problem.inner_product[component]) == 1
                inner_product[component] = self.truth_problem.inner_product[component][0]
            kwargs["inner_product"] = inner_product
        else:
            assert isinstance(kwargs["inner_product"], dict)
            assert set(kwargs["inner_product"].keys()) == set(kwargs["components"])
        #
        return (kwargs["components"], kwargs["inner_product"])
                
    # Compute the error of the reduced order output with respect to the full order one
    # for the current value of mu
    def compute_error_output(self, N=None, **kwargs):
        if self._compute_error__previous_mu != self.mu:
            self.truth_problem.solve(**kwargs)
            # Do not carry out truth solves anymore for the same parameter
            self._compute_error__previous_mu = self.mu
        if self._compute_error_output__previous_mu != self.mu:
            self.truth_problem.output()
            # Do not carry out truth solves anymore for the same parameter
            self._compute_error_output__previous_mu = self.mu
        # Compute the error on the output
        self.solve(N, **kwargs)
        self.output()
        return self._compute_error_output(**kwargs)
                
    # Internal method for output error computation
    def _compute_error_output(self, **kwargs):
        # Skip if no output defined
        if self._output is NotImplemented:
            assert self.truth_problem._output is NotImplemented
            return NotImplemented
        else: # Compute the error on the output
            reduced_output = self._output
            truth_output = self.truth_problem._output
            error_output = abs(truth_output - reduced_output)
            return error_output
        
    # Compute the relative error of the reduced order approximation with respect to the full order one
    # for the current value of mu
    def compute_relative_error_output(self, N=None, **kwargs):
        absolute_error_output = self.compute_error_output(N, **kwargs)
        return self._compute_relative_error_output(absolute_error_output, **kwargs)
        
    # Internal method for output error computation
    def _compute_relative_error_output(self, absolute_error_output, **kwargs):
        # Skip if no output defined
        if self._output is NotImplemented:
            assert self.truth_problem._output is NotImplemented
            assert absolute_error_output is NotImplemented
            return NotImplemented
        else: # Compute the relative error on the output
            truth_output = self.truth_problem._output
            return absolute_error_output/truth_output
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
        
    ## Export solution to file
    @override
    def export_solution(self, folder, filename, solution=None, component=None):
        if solution is None:
            solution = self._solution
        N = solution.N
        self.truth_problem.export_solution(folder, filename, self.Z[:N]*solution, component)
            
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{

    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        return self.truth_problem.compute_theta(term)
        
    ## Assemble the reduced order affine expansion
    def assemble_operator(self, term, current_stage="online"):
        assert current_stage in ("online", "offline")
        if current_stage == "online": # load from file
            if term in self.terms and not term in self.operator:
                self.operator[term] = OnlineAffineExpansionStorage(0) # it will be resized by load
            elif term.startswith("inner_product"):
                component = term.replace("inner_product", "").replace("_", "")
                if component != "":
                    assert component in self.components
                    self.inner_product[component] = OnlineAffineExpansionStorage(0) # it will be resized by load
                else:
                    assert len(self.components) == 1
                    self.inner_product = OnlineAffineExpansionStorage(0) # it will be resized by load
            # Note that it would not be needed to return the loaded operator in 
            # init(), since it has been already modified in-place. We do this, however,
            # because we want this interface to be compatible with the one in 
            # EllipticCoerciveProblem, i.e. we would like to be able to use a reduced 
            # problem also as a truth problem for a nested reduction
            if term in self.terms:
                self.operator[term].load(self.folder["reduced_operators"], "operator_" + term)
                return self.operator[term]
            elif term.startswith("inner_product"):
                if component != "":
                    self.inner_product[component].load(self.folder["reduced_operators"], term)
                    return self.inner_product[component]
                else:
                    self.inner_product.load(self.folder["reduced_operators"], term)
                    return self.inner_product
            elif term.startswith("dirichlet_bc"):
                raise ValueError("There should be no need to assemble Dirichlet BCs when querying online reduced problems.")
            else:
                raise ValueError("Invalid term for assemble_operator().")
        elif current_stage == "offline":
            # As in the previous case, there is no need to return anything because 
            # we are still training the reduced order model, so the previous remark 
            # (on the usage of a reduced problem as a truth one) cannot hold here.
            # However, in order to have a consistent interface we return the assembled
            # operator
            if term in self.terms:
                for q in range(self.Q[term]):
                    assert self.terms_order[term] in (0, 1, 2)
                    if self.terms_order[term] == 2:
                        self.operator[term][q] = transpose(self.Z)*self.truth_problem.operator[term][q]*self.Z
                    elif self.terms_order[term] == 1:
                        self.operator[term][q] = transpose(self.Z)*self.truth_problem.operator[term][q]
                    elif self.terms_order[term] == 0:
                        self.operator[term][q] = self.truth_problem.operator[term][q]
                    else:
                        raise AssertionError("Invalid value for order of term " + term)
                self.operator[term].save(self.folder["reduced_operators"], "operator_" + term)
                return self.operator[term]
            elif term.startswith("inner_product"):
                component = term.replace("inner_product", "").replace("_", "")
                if component != "":
                    assert component in self.components
                    assert len(self.inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                    assert len(self.truth_problem.inner_product[component]) == 1 # the affine expansion storage contains only the inner product matrix
                    self.inner_product[component][0] = transpose(self.Z)*self.truth_problem.inner_product[component][0]*self.Z
                    self.inner_product[component].save(self.folder["reduced_operators"], term)
                    return self.inner_product[component]
                else:
                    assert len(self.components) == 1 # single component case
                    assert len(self.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                    assert len(self.truth_problem.inner_product) == 1 # the affine expansion storage contains only the inner product matrix
                    self.inner_product[0] = transpose(self.Z)*self.truth_problem.inner_product[0]*self.Z
                    self.inner_product.save(self.folder["reduced_operators"], term)
                    return self.inner_product
            elif term.startswith("dirichlet_bc"):
                component = term.replace("dirichlet_bc", "").replace("_", "")
                if component != "":
                    assert component in self.components
                    has_non_homogeneous_dirichlet_bc = self.dirichlet_bc[component] and not self.dirichlet_bc_are_homogeneous[component]
                else:
                    assert len(self.components) == 1
                    component = None
                    has_non_homogeneous_dirichlet_bc = self.dirichlet_bc and not self.dirichlet_bc_are_homogeneous
                if has_non_homogeneous_dirichlet_bc:
                    # Compute lifting functions for the value of mu possibly provided by the user
                    theta_bc = self.compute_theta(term)
                    Q_dirichlet_bcs = len(theta_bc)
                    # Temporarily override compute_theta method to return only one nonzero 
                    # theta term related to boundary conditions
                    standard_compute_theta = self.truth_problem.compute_theta
                    for i in range(Q_dirichlet_bcs):
                        def modified_compute_theta(self, term_):
                            if term_ == term:
                                modified_theta_bc = list()
                                for j in range(Q_dirichlet_bcs):
                                    if j != i:
                                        modified_theta_bc.append(0.)
                                    else:
                                        modified_theta_bc.append(theta_bc[i])
                                return tuple(modified_theta_bc)
                            else:
                                return standard_compute_theta(term_)
                        self.truth_problem.compute_theta = types.MethodType(modified_compute_theta, self.truth_problem)
                        # ... and store the solution of the truth problem corresponding to that boundary condition
                        # as lifting function
                        solve_message = "Computing and storing lifting function n. " + str(i)
                        if component is not None:
                            solve_message += " for component " + component
                        solve_message += " (obtained for mu = " + str(self.mu) + ") in the basis matrix"
                        print(solve_message)
                        lifting = self.truth_problem.solve()
                        lifting /= theta_bc[i]
                        self.Z.enrich(lifting, component=component)
                    # Restore the standard compute_theta method
                    self.truth_problem.compute_theta = standard_compute_theta
            else:
                raise ValueError("Invalid term for assemble_operator().")
        else:
            raise AssertionError("Invalid stage in assemble_operator().")
    
    ## Return a lower bound for the coercivity constant
    def get_stability_factor(self):
        return self.truth_problem.get_stability_factor()
                    
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
