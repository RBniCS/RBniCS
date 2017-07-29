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

from dolfin import *
from rbnics import *
from utils import *

@ExactParametrizedFunctions()
class FitzHughNagumo(NonlinearParabolicProblem):
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        NonlinearParabolicProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.du = TrialFunction(V)
        (self.du1, self.du2) = split(self.du)
        self.u = self._solution
        (self.u1, self.u2) = split(self.u)
        self.v = TestFunction(V)
        (self.v1, self.v2) = split(self.v)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Problem coefficients
        self.epsilon = 0.015
        self.b = 0.5
        self.gamma = 2
        self.c = 0.05
        self.i0 = lambda t: 50000*t**3*exp(-15*t)
        self.f = lambda v: v*(v - 0.1)*(1 - v)
        # Customize time stepping parameters
        self._time_stepping_parameters.update({
            "report": True,
            "snes_solver": {
                "linear_solver": "mumps",
                "maximum_iterations": 20,
                "report": True
            }
        })
        
    ## Return custom problem name
    def name(self):
        return "FitzHughNagumoExact"
        
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "m":
            theta_m0 = self.epsilon
            theta_m1 = 1.
            return (theta_m0, theta_m1)
        elif term == "a" or term == "da":
            theta_a0 = self.epsilon**2
            theta_a1 = 1.
            theta_a2 = - self.b
            theta_a3 = self.gamma
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term == "f":
            t = self.t
            theta_f0 = self.c
            theta_f1 = self.epsilon**2*self.i0(t)
            return (theta_f0, theta_f1)
        else:
            raise ValueError("Invalid term for compute_theta().")
    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        (v1, v2) = (self.v1, self.v2)
        dx = self.dx
        if term == "m":
            (u1, u2) = (self.du1, self.du2)
            m0 = u1*v1*dx
            m1 = u2*v2*dx
            return (m0, m1)
        elif term == "a" or term == "da":
            (u1, u2) = (self.u1, self.u2)
            a0 = inner(grad(u1), grad(v1))*dx
            a1 = u2*v1*dx - self.f(u1)*v1*dx
            a2 = u1*v2*dx
            a3 = u2*v2*dx
            if term == "a":
                return (a0, a1, a2, a3)
            else:
                u = self.u
                du = self.du
                return tuple(derivative(ai, u, du) for ai in (a0, a1, a2, a3))
        elif term == "f":
            ds = self.ds
            f0 = v1*dx + v2*dx
            f1 = v1*ds(1)
            return (f0, f1)
        elif term == "inner_product":
            (u1, u2) = (self.du1, self.du2)
            x0 = inner(grad(u1),grad(v1))*dx + u2*v2*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
# Customize the resulting reduced problem
@CustomizeReducedProblemFor(NonlinearParabolicProblem)
def CustomizeReducedNonlinearParabolic(ReducedNonlinearParabolic_Base):
    class ReducedNonlinearParabolic(ReducedNonlinearParabolic_Base):
        def __init__(self, truth_problem, **kwargs):
            ReducedNonlinearParabolic_Base.__init__(self, truth_problem, **kwargs)
            self._time_stepping_parameters.update({
                "report": True,
                "nonlinear_solver": {
                    "report": True,
                    "line_search": "wolfe"
                }
            })
            
    return ReducedNonlinearParabolic

# 1. Read the mesh for this problem
mesh = Mesh("data/interval.xml")
subdomains = MeshFunction("size_t", mesh, "data/interval_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/interval_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)

# 3. Allocate an object of the FitzHughNagumo class
fitz_hugh_nagumo_problem = FitzHughNagumo(V, subdomains=subdomains, boundaries=boundaries)
mu_range = []
fitz_hugh_nagumo_problem.set_mu_range(mu_range)
fitz_hugh_nagumo_problem.set_time_step_size(0.02)
fitz_hugh_nagumo_problem.set_final_time(8)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(fitz_hugh_nagumo_problem)
pod_galerkin_method.set_Nmax(20)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(1)
reduced_fitz_hugh_nagumo_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
solution_over_time = fitz_hugh_nagumo_problem.solve()
fitz_hugh_nagumo_problem.export_solution("FitzHughNagumoExact", "offline_solution")
reduced_solution_over_time = reduced_fitz_hugh_nagumo_problem.solve()
reduced_fitz_hugh_nagumo_problem.export_solution("FitzHughNagumoExact", "online_solution")
Z = reduced_fitz_hugh_nagumo_problem.Z
plot_phase_space(solution_over_time, reduced_solution_over_time, Z, 0.0, "FitzHughNagumoExact", "phase_space_0.0")
plot_phase_space(solution_over_time, reduced_solution_over_time, Z, 0.1, "FitzHughNagumoExact", "phase_space_0.1")
plot_phase_space(solution_over_time, reduced_solution_over_time, Z, 0.5, "FitzHughNagumoExact", "phase_space_0.5")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(1)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
