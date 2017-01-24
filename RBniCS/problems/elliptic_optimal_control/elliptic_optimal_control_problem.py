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

from RBniCS.problems.base import ParametrizedDifferentialProblem
from RBniCS.backends import AffineExpansionStorage, Function, LinearSolver, product, sum, transpose
from RBniCS.utils.decorators import Extends, override

#~~~~~~~~~~~~~~~~~~~~~~~~~     ELLIPTIC COERCIVE PROBLEM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class EllipticCoerciveProblem
#
# Base class containing the definition of saddle point problems
@Extends(ParametrizedDifferentialProblem)
class EllipticOptimalControlProblem(ParametrizedDifferentialProblem):
    """
    The problem to be solved is 
        min {J(y, u) = 1/2 m(y - y_d, y - y_d) + 1/2 n(u, u)} 
        y in Y, u in U
        s.t.
        a(y, q) = c(u, q) + <f, q>      for all q in Q = Y
        
    This class will solve the following optimality conditions:
        m(y, z)           + a*(p, z) = <g, z>     for all z in Y
                  n(u, v) - c*(p, v) = 0          for all v in U
        a(y, q) - c(u, q)            = <f, q>     for all q in Q
        
    and compute the cost functional
        J(y, u) = 1/2 m(y, y) + 1/2 n(u, u) - <g, y> + 1/2 h
        
    where
        a*(., .) is the adjoint of a
        c*(., .) is the adjoint of c
        <g, y> = m(y_d, y)
        h = m(y_d, y_d)
    """
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the elliptic problem
    #  @{
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedDifferentialProblem.__init__(self, V, **kwargs)
        
        # Form names for saddle point problems
        self.terms = ["a", "a*", "c", "c*", "m", "n", "f", "g", "h"]
        self.terms_order = {
            "a": 2, "a*": 2, 
            "c": 2, "c*": 2,
            "m": 2, "n": 2,
            "f": 1, "g": 1,
            "h": 0
        }
        self.components = ["y", "u", "p"]
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
        
    ## Perform a truth solve
    @override
    def solve(self, **kwargs):
        assembled_operator = dict()
        for term in self.terms:
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        assembled_dirichlet_bc = dict()
        for component in ("y", "p"):
            if self.dirichlet_bc[component] is not None:
                assembled_dirichlet_bc[component] = sum(product(self.compute_theta("dirichlet_bc_" + component), self.dirichlet_bc[component]))
        assert self.dirichlet_bc["u"] is None, "Control should not be constrained by Dirichlet BCs"
        if len(assembled_dirichlet_bc) == 0:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            (
                  assembled_operator["m"]                           + assembled_operator["a*"]
                                          + assembled_operator["n"] - assembled_operator["c*"]
                + assembled_operator["a"] - assembled_operator["c"]
            ),
            self._solution,
            (
                  assembled_operator["g"]
                
                + assembled_operator["f"]
            ),
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._solution
        
    ## Perform a truth evaluation of the cost functional
    @override
    def output(self):
        assembled_operator = dict()
        for term in ("m", "n", "g", "h"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        self._output = (
            0.5*(transpose(self._solution)*assembled_operator["m"]*self._solution) + 
            0.5*(transpose(self._solution)*assembled_operator["n"]*self._solution) - 
            transpose(assembled_operator["g"])*self._solution + 
            0.5*assembled_operator["h"]
        )
        return self._output
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
