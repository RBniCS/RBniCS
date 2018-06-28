# Copyright (C) 2015-2018 by the RBniCS authors
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

import os
import hashlib
from numpy import isclose
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import adjoint, AffineExpansionStorage, assign, copy, EigenSolver, export, Function, import_, product, sum
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters

class ParametrizedCoercivityConstantEigenProblem(ParametrizedProblem):

    # Default initialization of members
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
        self._eigenvector = Function(truth_problem.V)
        # I/O
        self.folder["cache"] = os.path.join(folder_prefix, "cache")
        def _eigenvalue_cache_key_generator(*args, **kwargs):
            return args
        def _eigenvalue_cache_import(filename):
            self.import_eigenvalue(self.folder["cache"], filename)
            return self._eigenvalue
        def _eigenvalue_cache_export(filename):
            self.export_eigenvalue(self.folder["cache"], filename)
        def _eigenvalue_cache_filename_generator(*args, **kwargs):
            return self._cache_file(args)
        self._eigenvalue_cache = Cache(
            "problems",
            key_generator=_eigenvalue_cache_key_generator,
            import_=_eigenvalue_cache_import,
            export=_eigenvalue_cache_export,
            filename_generator=_eigenvalue_cache_filename_generator
        )
        def _eigenvector_cache_key_generator(*args, **kwargs):
            return args
        def _eigenvector_cache_import(filename):
            self.import_eigenvector(self.folder["cache"], filename)
            return self._eigenvector
        def _eigenvector_cache_export(filename):
            self.export_eigenvector(self.folder["cache"], filename)
        def _eigenvector_cache_filename_generator(*args, **kwargs):
            return self._cache_file(args)
        self._eigenvector_cache = Cache(
            "problems",
            key_generator=_eigenvector_cache_key_generator,
            import_=_eigenvector_cache_import,
            export=_eigenvector_cache_export,
            filename_generator=_eigenvector_cache_filename_generator
        )
    
    def init(self):
        # Store the symmetric part of the required term
        if self.operator is None: # init was not called already
            if isinstance(self.term, tuple):
                forms = (self.truth_problem.assemble_operator(self.term[0])[self.term[1]], )
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
        cache_key = self._cache_key()
        try:
            self._eigenvalue = self._eigenvalue_cache[cache_key]
            assign(self._eigenvector, self._eigenvector_cache[cache_key])
        except KeyError:
            self._solve()
            self._eigenvalue_cache[cache_key] = self._eigenvalue
            self._eigenvector_cache[cache_key] = copy(self._eigenvector)
        return (self._eigenvalue, self._eigenvector)
        
    def _solve(self):
        if self.multiply_by_theta:
            assert isinstance(self.term, str) # method untested otherwise
            O = sum(product(self.truth_problem.compute_theta(self.term), self.operator))  # noqa
        else:
            assert isinstance(self.term, tuple) # method untested otherwise
            assert len(self.operator) == 1
            O = self.operator[0]  # noqa
        assert len(self.inner_product) == 1
        inner_product = self.inner_product[0]
        
        if self.truth_problem.dirichlet_bc is not None:
            dirichlet_bcs_sum = sum(product((0., )*len(self.truth_problem.dirichlet_bc), self.truth_problem.dirichlet_bc))
            eigensolver = EigenSolver(self.truth_problem.V, O, inner_product, dirichlet_bcs_sum)
        else:
            eigensolver = EigenSolver(self.truth_problem.V, O, inner_product)
        eigensolver_parameters = dict()
        eigensolver_parameters["problem_type"] = "gen_hermitian"
        assert self.spectrum is "largest" or self.spectrum is "smallest"
        eigensolver_parameters["spectrum"] = self.spectrum + " real"
        if self.eigensolver_parameters is not None:
            if "spectral_transform" in self.eigensolver_parameters and self.eigensolver_parameters["spectral_transform"] == "shift-and-invert":
                eigensolver_parameters["spectrum"] = "target real"
            eigensolver_parameters.update(self.eigensolver_parameters)
        eigensolver.set_parameters(eigensolver_parameters)
        eigensolver.solve(1)
        
        r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
        r_vector, c_vector = eigensolver.get_eigenvector(0) # real and complex part of the eigenvectors
        
        assert isclose(c, 0), "The required eigenvalue is not real"
        
        self._eigenvalue = r
        assign(self._eigenvector, r_vector)
        
    def _cache_key(self):
        if self.multiply_by_theta:
            return (self.mu, self.term, self.spectrum)
        else:
            return (self.term, self.spectrum)
            
    def _cache_file(self, cache_key):
        return hashlib.sha1(str(cache_key).encode("utf-8")).hexdigest()
        
    def export_eigenvalue(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "coercivity_constant"
        export([self._eigenvalue], folder, filename + "_eigenvalue")
        
    def export_eigenvector(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "coercivity_constant"
        export(self._eigenvector, folder, filename + "_eigenvector")
        
    def import_eigenvalue(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "coercivity_constant"
        eigenvalue_storage = [0.]
        import_(eigenvalue_storage, folder, filename + "_eigenvalue")
        assert len(eigenvalue_storage) == 1
        self._eigenvalue = eigenvalue_storage[0]
        
    def import_eigenvector(self, folder=None, filename=None):
        if folder is None:
            folder = self.folder_prefix
        if filename is None:
            filename = "coercivity_constant"
        import_(self._eigenvector, folder, filename + "_eigenvector")
