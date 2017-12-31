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

from rbnics.backends import adjoint
from rbnics.problems.base import DualProblem
from rbnics.problems.elliptic_coercive.elliptic_coercive_problem import EllipticCoerciveProblem

EllipticCoerciveProblem_Dual_Base = DualProblem(EllipticCoerciveProblem)

class EllipticCoerciveProblem_Dual(EllipticCoerciveProblem_Dual_Base):
        
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            return self.primal_problem.compute_theta("a")
        elif term == "f":
            return tuple(-t for t in self.primal_problem.compute_theta("s"))
        elif term == "dirichlet_bc":
            raise ValueError("Dual problem has homogeneous Dirichlet BC, so compute_theta(\"dirichlet_bc\") should never be called.")
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        if term == "a":
            return tuple(adjoint(f) for f in self.primal_problem.assemble_operator("a"))
        elif term == "f":
            return self.primal_problem.assemble_operator("s")
        elif term == "dirichlet_bc":
            # We abuse here the fact that no theta is specified, so it will automatically
            # replaced by a zero multiplicative term. In this way, even if primal BCs are
            # non homogeneous, dual BCs are homogeneous
            return self.primal_problem.assemble_operator("dirichlet_bc")
        elif term == "inner_product":
            return self.primal_problem.assemble_operator("inner_product")
        elif term == "projection_inner_product":
            return self.primal_problem.assemble_operator("projection_inner_product")
        else:
            raise ValueError("Invalid term for assemble_operator().")
            
    # Return a lower bound for the coercivity constant
    def get_stability_factor(self):
        return self.primal_problem.get_stability_factor()
