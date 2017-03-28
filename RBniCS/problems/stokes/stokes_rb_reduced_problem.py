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

from math import sqrt
from numpy import isclose
from RBniCS.problems.stokes.stokes_reduced_problem import StokesReducedProblem
from RBniCS.backends import product, sum, transpose
from RBniCS.utils.decorators import Extends, override, ReducedProblemFor
from RBniCS.problems.stokes.stokes_problem import StokesProblem
from RBniCS.problems.base import RBReducedProblem
from RBniCS.reduction_methods.stokes import StokesRBReduction

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE REDUCED ORDER MODEL BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class StokesReducedOrderModelBase
#

StokesRBReducedProblem_Base = RBReducedProblem(StokesReducedProblem)

# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@Extends(StokesRBReducedProblem_Base) # needs to be first in order to override for last the methods
@ReducedProblemFor(StokesProblem, StokesRBReduction)
class StokesRBReducedProblem(StokesRBReducedProblem_Base):
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members.
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call to parent
        StokesRBReducedProblem_Base.__init__(self, truth_problem, **kwargs)
        
        # Skip useless Riesz products
        self.riesz_terms = ["a", "b", "bt", "f", "g"]
        self.riesz_product_terms = [("f", "f"), ("g", "g"), ("a", "f"), ("bt", "f"), ("b", "g"), ("a", "a"), ("a", "bt"), ("bt", "bt"), ("b", "b")]
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     ONLINE STAGE     ########################### 
    ## @defgroup OnlineStage Methods related to the online stage
    #  @{
    
    ## Return an error bound for the current solution
    @override
    def estimate_error(self):
        eps2 = self.get_residual_norm_squared()
        alpha = self.get_stability_factor()
        assert eps2 >= 0. or isclose(eps2, 0.)
        assert alpha >= 0.
        return sqrt(abs(eps2)/alpha)
        
    ## Return a relative error bound for the current solution
    @override
    def estimate_relative_error(self):
        return NotImplemented
    
    ## Return an error bound for the current output
    @override
    def estimate_error_output(self):
        return NotImplemented
        
    ## Return a relative error bound for the current output
    @override
    def estimate_relative_error_output(self):
        return NotImplemented
        
    ## Return the numerator of the error bound for the current solution
    def get_residual_norm_squared(self):
        N = self._solution.N
        theta_a = self.compute_theta("a")
        theta_b = self.compute_theta("b")
        theta_bt = self.compute_theta("bt")
        theta_f = self.compute_theta("f")
        theta_g = self.compute_theta("g")
        return (
              sum(product(theta_f, self.riesz_product["f", "f"], theta_f))
            + sum(product(theta_g, self.riesz_product["g", "g"], theta_g))
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.riesz_product["a", "f"][:N], theta_f)))
            + 2.0*(transpose(self._solution)*sum(product(theta_bt, self.riesz_product["bt", "f"][:N], theta_f)))
            + 2.0*(transpose(self._solution)*sum(product(theta_b, self.riesz_product["b", "g"][:N], theta_g)))
            + transpose(self._solution)*sum(product(theta_a, self.riesz_product["a", "a"][:N, :N], theta_a))*self._solution
            + 2.0*(transpose(self._solution)*sum(product(theta_a, self.riesz_product["a", "bt"][:N, :N], theta_bt))*self._solution)
            + transpose(self._solution)*sum(product(theta_bt, self.riesz_product["bt", "bt"][:N, :N], theta_bt))*self._solution
            + transpose(self._solution)*sum(product(theta_b, self.riesz_product["b", "b"][:N, :N], theta_b))*self._solution
        )
        
    # Custom Riesz solve inner product *not* to add inner product corresponding to supremizers
    def _init_riesz_solve_inner_product(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["u", "p"]
        # Call Parent
        StokesRBReducedProblem_Base._init_riesz_solve_inner_product(self)
        # Restore
        self.components = components_bak
        
    # Custom Riesz solve Dirichlet BCs *not* to add BCs corresponding to supremizers
    def _init_riesz_solve_homogeneous_dirichlet_bc(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["u", "p"]
        # Call Parent
        StokesRBReducedProblem_Base._init_riesz_solve_homogeneous_dirichlet_bc(self)
        # Restore
        self.components = components_bak
        
    #  @}
    ########################### end - ONLINE STAGE - end ########################### 
        
