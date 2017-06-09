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

from rbnics.problems.base import ParametrizedDifferentialProblem
from rbnics.backends import Function, LinearSolver, product, sum, transpose
from rbnics.utils.decorators import Extends, override

@Extends(ParametrizedDifferentialProblem)
class StokesOptimalControlProblem(ParametrizedDifferentialProblem):
    """
    The problem to be solved is 
        min {J(y, u) = 1/2 m(v - v_d, v - v_d) + 1/2 n(u, u)}
        y = (v, p) in Y = (V, P), u in U
        s.t.
        a(v, phi) + b(phi, p) = c(u, phi) + <f, phi>    for all phi in V
        b(v, xi)              = <l, xi>                 for all xi  in P
        
    This class will solve the following optimality conditions:
        m(v, psi)                         + a*(psi, w) + b*(psi, q) = <g, psi>     for all psi in V
                                            b*(w, pi )              = 0            for all pi  in P
                                n(u, tau) - c*(tau, w)              = 0            for all tau in U
        a(v, phi) + b(phi, p) - c(u, phi)                           = <f, phi>     for all phi in V
        b(v, xi)                                                    = <l, xi>      for all xi  in P
        
    and compute the cost functional
        J(y, u) = 1/2 m(v, v) + 1/2 n(u, u) - <g, v> + 1/2 h
        
    where
        a*(., .) is the adjoint of a
        b*(., .) is the adjoint of b
        c*(., .) is the adjoint of c
        <g, v> = m(v_d, v)
        h = m(v_d, v_d)
    """
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParametrizedDifferentialProblem.__init__(self, V, **kwargs)
        
        # Form names for saddle point problems
        self.terms = [
            "a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n", "f", "g", "h", "l",
            # Auxiliary terms for supremizer enrichment
            "bt_restricted", "bt*_restricted"
        ]
        self.terms_order = {
            "a": 2, "a*": 2, 
            "b": 2, "b*": 2, 
            "bt": 2, "bt*": 2, 
            "c": 2, "c*": 2,
            "m": 2, "n": 2,
            "f": 1, "g": 1, "l": 1,
            "h": 0,
            # Auxiliary terms for supremizer enrichment
            "bt_restricted": 2,
            "bt*_restricted": 2
        }
        self.components = ["v", "s", "p", "u", "w", "r", "q"]
        
        # Auxiliary storage for supremizer enrichment, using a subspace of V
        self._state_supremizer   = Function(V, "s")
        self._adjoint_supremizer = Function(V, "r")
        
    ## Perform a truth solve
    @override
    def _solve(self, **kwargs):
        assembled_operator = dict()
        for term in ("a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n", "f", "g", "l"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        assembled_dirichlet_bc = dict()
        for component in ("v", "p", "w", "q"):
            if self.dirichlet_bc[component] is not None:
                assembled_dirichlet_bc[component] = sum(product(self.compute_theta("dirichlet_bc_" + component), self.dirichlet_bc[component]))
        assert self.dirichlet_bc["u"] is None, "Control should not be constrained by Dirichlet BCs"
        if len(assembled_dirichlet_bc) == 0:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            (
                  assembled_operator["m"]                                                      + assembled_operator["a*"] + assembled_operator["bt*"]
                                                                                               + assembled_operator["b*"]
                                                                     + assembled_operator["n"] - assembled_operator["c*"]
                + assembled_operator["a"] + assembled_operator["bt"] - assembled_operator["c"]
                + assembled_operator["b"]
            ),
            self._solution,
            (
                  assembled_operator["g"]
                
                
                + assembled_operator["f"]
                + assembled_operator["l"]
            ),
            assembled_dirichlet_bc
        )
        solver.solve()
        
    def solve_state_supremizer(self):
        assert len(self.inner_product["s"]) == 1 # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.inner_product["s"][0]
        assembled_operator_bt = sum(product(self.compute_theta("bt_restricted"), self.operator["bt_restricted"]))
        assembled_operator_rhs = assembled_operator_bt*self._solution
        if self.dirichlet_bc["s"] is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc_s"), self.dirichlet_bc["s"]))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator_lhs,
            self._state_supremizer,
            assembled_operator_rhs,
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._state_supremizer
        
    def solve_adjoint_supremizer(self):
        assert len(self.inner_product["r"]) == 1 # the affine expansion storage contains only the inner product matrix
        assembled_operator_lhs = self.inner_product["r"][0]
        assembled_operator_btstar = sum(product(self.compute_theta("bt*_restricted"), self.operator["bt*_restricted"]))
        assembled_operator_rhs = assembled_operator_btstar*self._solution
        if self.dirichlet_bc["r"] is not None:
            assembled_dirichlet_bc = sum(product(self.compute_theta("dirichlet_bc_r"), self.dirichlet_bc["r"]))
        else:
            assembled_dirichlet_bc = None
        solver = LinearSolver(
            assembled_operator_lhs,
            self._adjoint_supremizer,
            assembled_operator_rhs,
            assembled_dirichlet_bc
        )
        solver.solve()
        return self._adjoint_supremizer
        
    ## Perform a truth evaluation of the cost functional
    @override
    def _compute_output(self):
        assembled_operator = dict()
        for term in ("m", "n", "g", "h"):
            assembled_operator[term] = sum(product(self.compute_theta(term), self.operator[term]))
        self._output = (
            0.5*(transpose(self._solution)*assembled_operator["m"]*self._solution) + 
            0.5*(transpose(self._solution)*assembled_operator["n"]*self._solution) - 
            transpose(assembled_operator["g"])*self._solution + 
            0.5*assembled_operator["h"]
        )
        
    # Custom combination of inner products *not* to add inner product corresponding to supremizers
    def _combine_all_inner_products(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["v", "p", "u", "w", "q"]
        # Call Parent
        StokesOptimalControlProblem_Base._combine_all_inner_products(self)
        # Restore
        self.components = components_bak
        
    # Custom combination of inner products *not* to add projection inner product corresponding to supremizers
    def _combine_all_projection_inner_products(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["v", "p", "u", "w", "q"]
        # Call Parent
        StokesOptimalControlProblem_Base._combine_all_projection_inner_products(self)
        # Restore
        self.components = components_bak
        
    # Custom combination of Dirichlet BCs *not* to add BCs corresponding to supremizers
    def _combine_and_homogenize_all_dirichlet_bcs(self):
        # Temporarily change self.components
        components_bak = self.components
        self.components = ["v", "p", "u", "w", "q"]
        # Call Parent
        StokesOptimalControlProblem_Base._combine_and_homogenize_all_dirichlet_bcs(self)
        # Restore
        self.components = components_bak
    
