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

import hashlib
from numpy import isclose
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import adjoint, AffineExpansionStorage, assign, copy, EigenSolver, export, Function, import_, product, sum
from rbnics.utils.decorators import sync_setters, Extends, override
from rbnics.utils.mpi import log, PROGRESS

@Extends(ParametrizedProblem)
class ParametrizedCoercivityConstantEigenProblem(ParametrizedProblem):

    ## Default initialization of members
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, term, multiply_by_theta, spectrum, eigensolver_parameters, folder_prefix):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix) # this class does not export anything
        self.truth_problem = truth_problem
        
        # Matrices/vectors resulting from the truth discretization
        self.term = term
        assert isinstance(self.term, (tuple, str))
        if isinstance(self.term, tuple):
            assert len(self.term) == 2
            isinstance(self.term[0], str)
            isinstance(self.term[1], int)
        self.multiply_by_theta = multiply_by_theta
        assert isinstance(self.multiply_by_theta, bool)
        self.operator = None # AffineExpansionStorage
        self.inner_product = None # AffineExpansionStorage, even though it will contain only one matrix
        self.spectrum = spectrum
        self.eigensolver_parameters = eigensolver_parameters
        
        # Avoid useless computations
        self._eigenvalue = 0.
        self._eigenvalue_cache = dict()
        self._eigenvector = Function(truth_problem.V)
        self._eigenvector_cache = dict()
        self.folder["cache"] = folder_prefix + "/" + "cache"
        self.cache_config = config.get("problems", "cache")
    
    def init(self):
        # Store the symmetric part of the required term
        if self.operator is None: # init was not called already
            if isinstance(self.term, tuple):
                forms = (self.truth_problem.assemble_operator(self.term[0])[ self.term[1] ], )
            else:
                assert isinstance(self.term, str)
                forms = self.truth_problem.assemble_operator(self.term)
            self.operator = AffineExpansionStorage(tuple(0.5*(f + adjoint(f)) for f in forms))
        
        # Store the inner product matrix
        if self.inner_product is None: # init was not called already
            self.inner_product = AffineExpansionStorage(self.truth_problem.assemble_operator("inner_product"))
            
        # Also make sure to create folder for cache
        self.folder.create()
    
    def solve(self):
        (cache_key, cache_file) = self._cache_key_and_file()
        assert (
            (cache_key in self._eigenvalue_cache)
                ==
            (cache_key in self._eigenvector_cache)
        )
        if "RAM" in self.cache_config and cache_key in self._eigenvalue_cache: 
            log(PROGRESS, "Loading coercivity constant from cache")
            self._eigenvalue = self._eigenvalue_cache[cache_key]
            assign(self._eigenvector, self._eigenvector_cache[cache_key])
        elif "Disk" in self.cache_config and self.import_solution(self.folder["cache"], cache_file):
            log(PROGRESS, "Loading coercivity constant from file")
            if "RAM" in self.cache_config:
                self._eigenvalue_cache[cache_key] = self._eigenvalue
                self._eigenvector_cache[cache_key] = copy(self._eigenvector)
        else: # No precomputed solution available. Truth solve is performed.
            log(PROGRESS, "Solving coercivity constant eigenproblem")
            self._solve()
            if "RAM" in self.cache_config:
                self._eigenvalue_cache[cache_key] = self._eigenvalue
                self._eigenvector_cache[cache_key] = copy(self._eigenvector)
            self.export_solution(self.folder["cache"], cache_file) # Note that we export to file regardless of config options, because they may change across different runs
        return (self._eigenvalue, self._eigenvector)
        
    def _solve(self):
        if self.multiply_by_theta:
            assert isinstance(self.term, str) # method untested otherwise
            O = sum(product(self.truth_problem.compute_theta(self.term), self.operator))
        else:
            assert isinstance(self.term, tuple) # method untested otherwise
            assert len(self.operator) == 1
            O = self.operator[0]
        assert len(self.inner_product) == 1
        X = self.inner_product[0]
        
        if self.truth_problem.dirichlet_bc is not None:
            dirichlet_bcs_sum = sum(product((0., )*len(self.truth_problem.dirichlet_bc), self.truth_problem.dirichlet_bc))
            eigensolver = EigenSolver(self.truth_problem.V, O, X, dirichlet_bcs_sum)
        else:
            eigensolver = EigenSolver(self.truth_problem.V, O, X)
        eigensolver_parameters = dict()
        eigensolver_parameters["problem_type"] = "gen_hermitian"
        assert self.spectrum is "largest" or self.spectrum is "smallest"
        eigensolver_parameters["spectrum"] = self.spectrum + " real"
        if self.eigensolver_parameters is not None:
            eigensolver_parameters.update(self.eigensolver_parameters)
        eigensolver.set_parameters(eigensolver_parameters)
        eigensolver.solve(1)
        
        r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
        r_vector, c_vector = eigensolver.get_eigenvector(0) # real and complex part of the eigenvectors
        
        assert isclose(c, 0), "The required eigenvalue is not real"
        
        self._eigenvalue = r
        assign(self._eigenvector, r_vector)
            
    def _cache_key_and_file(self):
        if self.multiply_by_theta:
            cache_key = (self.mu, self.term, self.spectrum)
        else:
            cache_key = (self.term, self.spectrum)
        cache_file = hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        return (cache_key, cache_file)
        
    ## Export solution to file
    def export_solution(self, folder, filename):
        export([self._eigenvalue], folder, filename + "_eigenvalue")
        export(self._eigenvector, folder, filename + "_eigenvector")
        
    ## Import solution from file
    def import_solution(self, folder, filename):
        eigenvalue_storage = [0.]
        import_successful = import_(eigenvalue_storage, folder, filename + "_eigenvalue")
        if import_successful:
            assert len(eigenvalue_storage) == 1
            self._eigenvalue = eigenvalue_storage[0]
            import_successful = import_(self._eigenvector, folder, filename + "_eigenvector")
        return import_successful
        
