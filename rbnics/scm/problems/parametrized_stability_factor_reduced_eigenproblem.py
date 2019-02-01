# Copyright (C) 2015-2019 by the RBniCS authors
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

from numpy import isclose
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import assign, copy, product, sum
from rbnics.backends.online import OnlineEigenSolver, OnlineFunction
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import sync_setters

class ParametrizedStabilityFactorReducedEigenProblem(ParametrizedProblem):

    # Default initialization of members
    @sync_setters("reduced_problem", "set_mu", "mu")
    @sync_setters("reduced_problem", "set_mu_range", "mu_range")
    def __init__(self, reduced_problem, spectrum, eigensolver_parameters, folder_prefix):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix) # this class does not export anything
        self.reduced_problem = reduced_problem
        
        # Matrices/vectors resulting from the truth discretization
        self.operator = {
            "stability_factor_left_hand_matrix": None, # OnlineAffineExpansionStorage
            "stability_factor_right_hand_matrix": None # OnlineAffineExpansionStorage, even though it will contain only one matrix
        }
        self.spectrum = spectrum
        self.eigensolver_parameters = eigensolver_parameters
        
        # Solution
        self._eigenvalue = 0.
        self._eigenvector = None # OnlineFunction
        # I/O
        def _eigenvalue_cache_key_generator(*args, **kwargs):
            return args
        self._eigenvalue_cache = Cache(
            "reduced problems",
            key_generator=_eigenvalue_cache_key_generator
        )
        def _eigenvector_cache_key_generator(*args, **kwargs):
            return args
        self._eigenvector_cache = Cache(
            "reduced problems",
            key_generator=_eigenvector_cache_key_generator
        )
    
    def init(self, current_stage="online"):
        # Store the left and right hand side operators
        if self.operator["stability_factor_left_hand_matrix"] is None: # init was not called already
            self.operator["stability_factor_left_hand_matrix"] = self.reduced_problem.operator["stability_factor_left_hand_matrix"]
        if self.operator["stability_factor_right_hand_matrix"] is None: # init was not called already
            self.operator["stability_factor_right_hand_matrix"] = self.reduced_problem.operator["stability_factor_right_hand_matrix"]
            assert len(self.operator["stability_factor_right_hand_matrix"]) == 1
            
    def solve(self, N=None, **kwargs):
        N, kwargs = self.reduced_problem._online_size_from_kwargs(N, **kwargs)
        N += self.reduced_problem.N_bc
        self._eigenvector = OnlineFunction(N)
        cache_key = self._cache_key(**kwargs)
        try:
            self._eigenvalue = self._eigenvalue_cache[cache_key]
            assign(self._eigenvector, self._eigenvector_cache[cache_key])
        except KeyError:
            self._solve(N, **kwargs)
            self._eigenvalue_cache[cache_key] = self._eigenvalue
            self._eigenvector_cache[cache_key] = copy(self._eigenvector)
        return (self._eigenvalue, self._eigenvector)
        
    def _solve(self, N, **kwargs):
        assert self.operator["stability_factor_left_hand_matrix"] is not None
        A = sum(product(self.reduced_problem.compute_theta("stability_factor_left_hand_matrix"), self.operator["stability_factor_left_hand_matrix"][:N, :N]))
        assert self.operator["stability_factor_right_hand_matrix"] is not None
        assert len(self.operator["stability_factor_right_hand_matrix"]) == 1
        B = self.operator["stability_factor_right_hand_matrix"][0][:N, :N]
        
        eigensolver = OnlineEigenSolver(self.reduced_problem.stability_factor_basis_functions, A, B)
        eigensolver_parameters = dict()
        assert self.spectrum == "largest" or self.spectrum == "smallest"
        eigensolver_parameters["spectrum"] = self.spectrum + " real"
        eigensolver_parameters.update(self.eigensolver_parameters)
        eigensolver.set_parameters(eigensolver_parameters)
        eigensolver.solve(1)
        
        r, c = eigensolver.get_eigenvalue(0) # real and complex part of the eigenvalue
        r_vector, c_vector = eigensolver.get_eigenvector(0) # real and complex part of the eigenvectors
        
        assert isclose(c, 0.), "The required eigenvalue is not real"
        
        self._eigenvalue = r
        assign(self._eigenvector, r_vector)
        
    def _cache_key(self, **kwargs):
        return (self.mu, self.spectrum, tuple(sorted(kwargs.items())))
