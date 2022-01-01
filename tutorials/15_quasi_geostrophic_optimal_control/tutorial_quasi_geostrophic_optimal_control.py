# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from rbnics.backends import export, import_
from problems import *
from reduction_methods import *


class GeostrophicOptimalControl(GeostrophicOptimalControlProblem):
    def __init__(self, W, **kwargs):
        GeostrophicOptimalControlProblem.__init__(self, W, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        trial = TrialFunction(W)
        (self.ypsi, self.yq, self.u, self.ppsi, self.pq) = split(trial)
        test = TestFunction(W)
        (self.zpsi, self.zq, self.v, self.qpsi, self.qq) = split(test)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.f = Expression("-sin(pi*x[1])", element=W.sub(0).ufl_element())
        # Regularization coefficient
        self.alpha = 1.e-5
        self.yd = self._compute_yd()

    def _compute_yd(self):
        """
        The desired state is (a component of) the solution of a nonlinear geostrophic problem for fixed parameters
        """
        # State space
        state_element = MixedElement(scalar_element, scalar_element)
        W = FunctionSpace(mesh, state_element, components=["psi", "q"])
        # Solution
        psiq = Function(W)
        # Import solution from file, if possible
        try:
            import_(psiq, self.name(), "yd")
        except OSError:
            # Fixed problem coefficients (mu[0] is (delta_M/L)**3, mu[1] is C)
            delta_M = 7e4
            L = 1e6
            C = 0
            # Fixed problem coefficients related to the nonlinear term
            delta_I = 7e4
            # Test and trial functions for variational forms definition
            phip = TestFunction(W)
            (phi, p) = split(phip)
            delta_psiq = TrialFunction(W)
            (delta_psi, delta_q) = split(delta_psiq)
            (psi, q) = split(psiq)
            # Variational forms
            F = (inner(q, phi) * dx + inner(grad(psi), grad(phi)) * dx
                 + Constant(- (delta_I / L)**2) * inner(psi, q.dx(1) * p.dx(0) - q.dx(0) * p.dx(1)) * dx
                 + inner(psi.dx(0), p) * dx + Constant((delta_M / L)**3) * inner(grad(q), grad(p)) * dx
                 + Constant(C) * inner(q, p) * dx
                 - inner(self.f, p) * dx)
            J = derivative(F, psiq, delta_psiq)
            # Boundary conditions
            bc = [DirichletBC(W, Constant((0., 0.)), boundaries, idx) for idx in [1, 2, 3, 4]]
            # Solve nonlinear problem
            snes_solver_parameters = {"nonlinear_solver": "snes",
                                      "snes_solver": {"linear_solver": "mumps",
                                                      "maximum_iterations": 20,
                                                      "report": True}}
            problem = NonlinearVariationalProblem(F, psiq, bc, J)
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(snes_solver_parameters)
            solver.solve()
            # Export solution to file
            export(psiq, self.name(), "yd")
        # Tracking is on the psi component
        (psi, q) = psiq.split(deepcopy=True)
        return psi

    def compute_theta(self, term):
        mu = self.mu
        if term == "a" or term == "a*":
            theta_a0 = 1.0
            theta_a1 = mu[0]
            theta_a2 = mu[1]
            theta_a3 = 1.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term == "c" or term == "c*":
            theta_c0 = 1.0
            return (theta_c0,)
        elif term == "m":
            theta_m0 = 1.0
            return (theta_m0,)
        elif term == "n":
            alpha = self.alpha
            theta_n0 = alpha
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0,)
        elif term == "h":
            theta_h0 = 1.0
            return (theta_h0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            yq = self.yq
            ypsi = self.ypsi
            qpsi = self.qpsi
            qq = self.qq
            a0 = inner(grad(ypsi), grad(qpsi)) * dx + yq * qpsi * dx
            a1 = inner(grad(yq), grad(qq)) * dx
            a2 = inner(yq, qq) * dx
            a3 = ypsi.dx(0) * qq * dx
            return (a0, a1, a2, a3)
        elif term == "a*":
            pq = self.pq
            zpsi = self.zpsi
            zq = self.zq
            ppsi = self.ppsi
            as0 = inner(grad(ppsi), grad(zpsi)) * dx + ppsi * zq * dx
            as1 = inner(grad(pq), grad(zq)) * dx
            as2 = inner(pq, zq) * dx
            as3 = - pq.dx(0) * zpsi * dx
            return (as0, as1, as2, as3)
        elif term == "c":
            u = self.u
            qq = self.qq
            c0 = u * qq * dx
            return (c0,)
        elif term == "c*":
            v = self.v
            pq = self.pq
            cs0 = v * pq * dx
            return (cs0,)
        elif term == "m":
            ypsi = self.ypsi
            zpsi = self.zpsi
            m0 = ypsi * zpsi * dx
            return (m0,)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = u * v * dx
            return (n0,)
        elif term == "f":
            qq = self.qq
            f0 = Constant(0.0) * qq * dx
            return (f0,)
        elif term == "g":
            yd = self.yd
            zpsi = self.zpsi
            g0 = yd * zpsi * dx
            return (g0,)
        elif term == "h":
            yd = self.yd
            h0 = assemble(yd * yd * dx)
            return (h0,)
        elif term == "dirichlet_bc_ypsi":
            bc0 = [DirichletBC(self.V.sub(0), Constant(0.0), boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "dirichlet_bc_yq":
            bc0 = [DirichletBC(self.V.sub(1), Constant(0.0), self.boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "dirichlet_bc_ppsi":
            bc0 = [DirichletBC(self.V.sub(3), Constant(0.0), self.boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "dirichlet_bc_pq":
            bc0 = [DirichletBC(self.V.sub(4), Constant(0.0), self.boundaries, idx) for idx in [1, 2, 3, 4]]
            return (bc0,)
        elif term == "inner_product_ypsi":
            ypsi = self.ypsi
            zpsi = self.zpsi
            x0 = inner(grad(ypsi), grad(zpsi)) * dx
            return (x0,)
        elif term == "inner_product_yq":
            yq = self.yq
            zq = self.zq
            x0 = inner(grad(yq), grad(zq)) * dx
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = u * v * dx
            return (x0,)
        elif term == "inner_product_ppsi":
            ppsi = self.ppsi
            qpsi = self.qpsi
            x0 = inner(grad(ppsi), grad(qpsi)) * dx
            return (x0,)
        elif term == "inner_product_pq":
            pq = self.pq
            qq = self.qq
            x0 = inner(grad(pq), grad(qq)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/square.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, five components)
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(scalar_element, scalar_element, scalar_element, scalar_element, scalar_element)
V = FunctionSpace(mesh, element, components=["ypsi", "yq", "u", "ppsi", "pq"])

# 3. Allocate an object of the Geostrophic class
problem = GeostrophicOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1e-4, 1.0), (1e-4, 1.0)]
problem.set_mu_range(mu_range)

# 4. Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(problem)
pod_galerkin_method.set_Nmax(10)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(50, sampling=LogUniformDistribution())
reduced_problem = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = ((7e4 / 1e6)**3, 1e-4)
reduced_problem.set_mu(online_mu)
reduced_problem.solve()
reduced_problem.export_solution("GeostrophicOptimalControl", "online_solution")

# 7. Perform an error analysis
pod_galerkin_method.initialize_testing_set(50, sampling=LogUniformDistribution())
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
