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

from rbnics.backends import adjoint, Function
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.utils.decorators import DualProblem, Extends, override

@Extends(EllipticCoerciveProblem)
@DualProblem
class EllipticCoerciveProblem_Dual(EllipticCoerciveProblem):

    ## Default initialization of members.
    @override
    def __init__(self, primal_problem):
        EllipticCoerciveProblem.__init__(self, primal_problem.V)
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    @override
    def compute_theta(self, term):
        if term == "a":
            return self.primal_problem.compute_theta("a")
        elif term == "f":
            return tuple(-t for t in self.primal_problem.compute_theta("s"))
        elif term == "dirichlet_bc":
            raise ValueError("Dual problem has homogeneous Dirichlet BC, so compute_theta(\"dirichlet_bc\") should never be called.")
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    @override
    def assemble_operator(self, term):
        if term == "a":
            return tuple(adjoint(f) for f in self.primal_problem.assemble_operator("a"))
        elif term == "f":
            return self.primal_problem.assemble_operator("s")
        elif term == "dirichlet_bc":
            homogeneous_bc_expansion = list()
            for bc_list in self.primal_problem.assemble_operator("dirichlet_bc"):
                homogeneous_bc_list = list()
                for bc in bc_list:
                    homogeneous_bc_list.append(0.*bc)
                homogeneous_bc_expansion.append(homogeneous_bc_list)
            return tuple(homogeneous_bc_expansion)
        elif term == "inner_product":
            return self.primal_problem.assemble_operator("inner_product")
        elif term == "projection_inner_product":
            return self.primal_problem.assemble_operator("projection_inner_product")
        else:
            raise ValueError("Invalid term for assemble_operator().")
            
    ## Return a lower bound for the coercivity constant
    @override
    def get_stability_factor(self):
        return self.primal_problem.get_stability_factor()
        
