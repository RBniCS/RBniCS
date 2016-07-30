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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from dolfin import adjoint, Function, DirichletBC
from RBniCS.problems import ParametrizedProblem
from RBniCS.linear_algebra import AffineExpansionOfflineStorage, sum, product, TruthEigenSolver
from RBniCS.utils.decorators import SyncSetters, extends, override

@extends(ParametrizedProblem) # needs to be first in order to override for last the methods
@SyncSetters("truth_problem", "set_mu", "mu")
@SyncSetters("truth_problem", "set_mu_range", "mu_range")
class ParametrizedHermitianEigenProblem(ParametrizedProblem):
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the EIM object
    #  @{

    ## Default initialization of members
    @override
    def __init__(self, truth_problem, term, multiply_by_theta, constrain_eigenvalue, spectrum, eigensolver_parameters):
        # Call the parent initialization
        ParametrizedProblem.__init__(self, folder_prefix="") # this class does not export anything
        self.truth_problem = truth_problem
        
        # We need to discard dofs related to bcs in eigenvalue computations. To avoid having to create a PETSc submatrix
        # we simply zero rows and columns and replace the diagonal element with an eigenvalue that for sure
        # will not be the one we are interested in
        self.constrain_eigenvalue = constrain_eigenvalue
        # Matrices/vectors resulting from the truth discretization: condensed version discard
        # Dirichlet DOFs
        self.term = term
        assert isinstance(self.term, tuple) or isinstance(self.term, str)
        if isinstance(self.term, tuple):
            assert len(self.term) == 2
            isinstance(self.term[0], str)
            isinstance(self.term[1], int)
        self.multiply_by_theta = multiply_by_theta
        assert isinstance(self.multiply_by_theta, bool)
        self.operator__condensed = AffineExpansionOfflineStorage()
        self.inner_product__condensed = AffineExpansionOfflineStorage() # even though it will contain only one matrix
        self.spectrum = spectrum
        self.eigensolver_parameters = eigensolver_parameters
        
        # Avoid useless computations
        self.solve.__func__.previous_mu = None
        self.solve.__func__.previous_eigenvalue = None
        self.solve.__func__.previous_eigenvector = None
        
    #  @}
    ########################### end - CONSTRUCTORS - end ###########################
    
    def init(self):
        # Condense the symmetric part of the required term
        if isinstance(self.term, tuple):
            forms = (self.truth_problem.assemble_operator(self.term[0])[ self.term[1] ], )
        else:
            assert isinstance(self.term, str)
            forms = self.truth_problem.assemble_operator(self.term)
        symmetric_forms = [ 0.5*(form + adjoint(form)) for form in forms]
        self.operator__condensed.init(symmetric_forms)
        self.clear_constrained_dofs(self.operator__condensed, self.constrain_eigenvalue)
        
        # Condense the inner product matrix
        self.inner_product__condensed.init(self.truth_problem.assemble_operator("inner_product"))
        self.clear_constrained_dofs(self.inner_product__condensed, 1.)
        
    # Clear constrained dofs
    def clear_constrained_dofs(self, operator, diag_value):
        dirichlet_bc = self.truth_problem.dirichlet_bc
        V = self.truth_problem.V
        for q in range(len(operator)):
            if len(dirichlet_bc) > 0:
                dummy = Function(V)
                for q in range(len(dirichlet_bc)):
                    for i in range(len(dirichlet_bc[q])):
                        dirichlet_bc_q_i = DirichletBC(*dirichlet_bc[q][i])
                        dirichlet_bc_q_i.zero(operator[q])
                        dirichlet_bc_q_i.zero_columns(operator[q], dummy.vector(), diag_value)
    
    def solve(self):
        if self.solve.__func__.previous_mu == self.mu:
            return (self.solve.__func__.previous_eigenvalue, self.solve.__func__.previous_eigenvector)
        else:
            if self.multiply_by_theta:
                assert isinstance(self.term, str) # method untested otherwise
                O = sum(product(self.truth_problem.compute_theta(self.term), self.operator__condensed))
            else:
                assert isinstance(self.term, tuple) # method untested otherwise
                theta = (1.,)
                assert len(theta) == len(self.operator__condensed)
                O = sum(product(theta, self.operator__condensed))
            assert len(self.inner_product__condensed) == 1
            X = self.inner_product__condensed[0]
            
            eigensolver = TruthEigenSolver(O, X)
            eigensolver.parameters["problem_type"] = "gen_hermitian"
            assert self.spectrum is "largest" or self.spectrum is "smallest"
            eigensolver.parameters["spectrum"] = self.spectrum + " real"
            if self.eigensolver_parameters is not None:
                eigensolver.parameters.update(self.eigensolver_parameters)
            eigensolver.solve(1)
            
            r, c, r_vector, c_vector = eigensolver.get_eigenpair(0) # real and complex part of the (eigenvalue, eigenvectors)
            
            from numpy import isclose
            assert isclose(c, 0), "The required eigenvalue is not real"
            assert not isclose(r, self.constrain_eigenvalue), "The required eigenvalue is too close to the one used to constrain Dirichlet boundary conditions"
            #assert r >= 0 or isclose(r, 0), "The required eigenvalue is not positive"
            
            self.solve.__func__.previous_mu = self.mu
            self.solve.__func__.previous_eigenvalue = r
            self.solve.__func__.previous_eigenvector = r_vector
            
            return (r, r_vector)
        
