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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import isclose
from rbnics.problems.base import ParametrizedProblem
from rbnics.backends import adjoint, AffineExpansionStorage, EigenSolver, sum, product
from rbnics.utils.decorators import sync_setters, Extends, override

@Extends(ParametrizedProblem)
class ParametrizedCoercivityConstantEigenProblem(ParametrizedProblem):
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{

    ## Default initialization of members
    @override
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, term, multiply_by_theta, spectrum, eigensolver_parameters):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix="") # this class does not export anything
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
        self.operator = AffineExpansionStorage()
        self.inner_product = AffineExpansionStorage() # even though it will contain only one matrix
        self.spectrum = spectrum
        self.eigensolver_parameters = eigensolver_parameters
        
        # Avoid useless computations
        self._solve__previous_mu = None
        self._solve__previous_eigenvalue = None
        self._solve__previous_eigenvector = None
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    def init(self):
        # Store the symmetric part of the required term
        if isinstance(self.term, tuple):
            forms = (self.truth_problem.assemble_operator(self.term[0])[ self.term[1] ], )
        else:
            assert isinstance(self.term, str)
            forms = self.truth_problem.assemble_operator(self.term)
        self.operator = AffineExpansionStorage(tuple(0.5*(f + adjoint(f)) for f in forms))
        
        # Store the inner product matrix
        self.inner_product = AffineExpansionStorage(self.truth_problem.assemble_operator("inner_product"))
    
    def solve(self):
        if self._solve__previous_mu == self.mu:
            return (self._solve__previous_eigenvalue, self._solve__previous_eigenvector)
        else:
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
                eigensolver = EigenSolver(self.truth_problem.V, O, X, self.truth_problem.dirichlet_bc)
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
            
            self._solve__previous_mu = self.mu
            self._solve__previous_eigenvalue = r
            self._solve__previous_eigenvector = r_vector
            
            return (r, r_vector)
        
