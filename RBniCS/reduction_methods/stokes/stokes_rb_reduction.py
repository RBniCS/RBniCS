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
## @file stokes_pod_galerkin_reduction.py
#  @brief Implementation of a POD-Galerkin ROM for elliptic coervice problems
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends import GramSchmidt
from RBniCS.utils.decorators import Extends, override, ReductionMethodFor
from RBniCS.problems.stokes.stokes_problem import StokesProblem
from RBniCS.reduction_methods.base import RBReduction
from RBniCS.reduction_methods.stokes.stokes_reduction_method import StokesReductionMethod

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE POD BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class StokesRBReduction
#

StokesRBReduction_Base = RBReduction(StokesReductionMethod)

# Base class containing the interface of a POD-Galerkin ROM
# for elliptic coercive problems
@Extends(StokesRBReduction_Base) # needs to be first in order to override for last the methods
@ReductionMethodFor(StokesProblem, "ReducedBasis")
class StokesRBReduction(StokesRBReduction_Base):    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the POD-Galerkin ROM object
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, truth_problem, **kwargs):
        # Call the parent initialization
        StokesRBReduction_Base.__init__(self, truth_problem, **kwargs)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ## Initialize data structures required for the offline phase: overridden version because supremizer GS is different from a standard component
    @override
    def _init_offline(self):
        # We cannot use the standard initialization provided by RBReduction because
        # supremizer GS requires a custom initialization. We thus duplicate here part of its code
        
        # Call parent of parent (!) to initialize inner product and reduced problem
        output = StokesReductionMethod._init_offline(self)
        
        # Declare a new GS for each basis component
        self.GS = dict()
        for component in ("u", "p"):
            inner_product = self.truth_problem.inner_product[component][0]
            self.GS[component] = GramSchmidt(inner_product)
        for component in ("s", ):
            inner_product = self.truth_problem.inner_product["u"][0] # instead of the one for "s", which has smaller size
            self.GS[component] = GramSchmidt(inner_product)
            
        # Return
        return output
    
    ## Update the basis matrix: overridden version because the input argument now contains both snapshot and supremizer
    def update_basis_matrix(self, snapshot_and_supremizer):
        assert isinstance(snapshot_and_supremizer, tuple)
        assert len(snapshot_and_supremizer) == 2
        snapshot = snapshot_and_supremizer[0]
        supremizer = snapshot_and_supremizer[1]
        for component in ("u", "s", "p"):
            if component == "s":
                self.reduced_problem.Z.enrich(supremizer, component=component)
            else:
                self.reduced_problem.Z.enrich(snapshot, component=component)
            self.GS[component].apply(self.reduced_problem.Z[component], self.reduced_problem.N_bc[component])
            self.reduced_problem.N[component] += 1
            
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
