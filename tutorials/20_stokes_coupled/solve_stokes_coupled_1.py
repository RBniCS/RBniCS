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
## @file solve_elast.py
#  @brief Example 2: elastic block test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *
from sampling import LinearlyDependentUniformDistribution

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: STOKES CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
shape_parametrization = (
    ("mu[4]*x[0] + mu[1] - mu[4]", "tan(mu[5])*x[0] + mu[0]*x[1] + mu[2] - tan(mu[5]) - mu[0]"), # subdomain 1
    ("mu[1]*x[0]", "mu[3]*x[1] + mu[2] + mu[0] - 2*mu[3]"), # subdomain 2
    ("mu[1]*x[0]", "mu[0]*x[1] + mu[2] - mu[0]"), # subdomain 3
    ("mu[1]*x[0]", "mu[2]*x[1]"), # subdomain 4
)

@EIM()
@ShapeParametrization(*shape_parametrization)
class Stokes(StokesProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        StokesProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        up = TrialFunction(V)
        (self.u, self.p) = split(up)
        vq = TestFunction(V)
        (self.v, self.q) = split(vq)
        self.s = TrialFunction(V.sub(0).collapse())
        self.r = TestFunction(V.sub(0).collapse())
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        #
        self.f = Constant((0.0, -10.0))
        self.g = Constant(0.0)
        # Store parametrized tensors related to shape parametrization
        expression_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        scalar_element = V.sub(0).sub(0).ufl_element()
        tensor_element = TensorElement(scalar_element)
        det_deformation_gradient = (
            "mu[4]*mu[0]", # subdomain 1
            "mu[1]*mu[3]", # subdomain 2
            "mu[1]*mu[0]", # subdomain 3
            "mu[1]*mu[2]"  # subdomain 4
        )
        tensor_kappa = (
            (("mu[0]/mu[4]", "-tan(mu[5])/mu[4]"), ("-tan(mu[5])/mu[4]", "(pow(tan(mu[5]), 2) + pow(mu[4], 2))/(mu[4]*mu[0])")), # subdomain 1
            (("mu[3]/mu[1]", "0."), ("0.", "mu[1]/mu[3]")), # subdomain 2
            (("mu[0]/mu[1]", "0."), ("0.", "mu[1]/mu[0]")), # subdomain 3
            (("mu[2]/mu[1]", "0."), ("0.", "mu[1]/mu[2]"))  # subdomain 4
        )
        tensor_chi = (
            (("mu[0]", "0."), ("-tan(mu[5])", "mu[4]")), # subdomain 1
            (("mu[3]", "0."), ("0.", "mu[1]")), # subdomain 2
            (("mu[0]", "0."), ("0.", "mu[1]")), # subdomain 3
            (("mu[2]", "0."), ("0.", "mu[1]"))  # subdomain 4
        )
        self.det_deformation_gradient = list()
        self.tensor_kappa = list()
        self.tensor_chi = list()
        for s in range(4):
            self.det_deformation_gradient.append(ParametrizedExpression(self, det_deformation_gradient[s], mu=expression_mu, element=scalar_element))
            self.tensor_kappa.append(ParametrizedExpression(self, tensor_kappa[s], mu=expression_mu, element=tensor_element))
            self.tensor_chi.append(ParametrizedExpression(self, tensor_chi[s], mu=expression_mu, element=tensor_element))
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        mu4 = mu[3]
        mu5 = mu[4]
        mu6 = mu[5]
        if term == "a":
            theta_a0 = 1.
            return (theta_a0,)
        elif term == "b" or term == "bt" or term == "bt_restricted":
            theta_b0 = 1.
            return (theta_b0,)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        if term == "a":
            u = self.u
            v = self.v
            tensor_kappa = self.tensor_kappa
            a0 = 0
            for s in range(4):
                a0 += inner(grad(u)*tensor_kappa[s], grad(v))*dx(s + 1)
            return (a0,)
        elif term == "b":
            u = self.u
            q = self.q
            tensor_chi = self.tensor_chi
            b0 = 0
            for s in range(4):
                b0 += - q*tr(tensor_chi[s]*grad(u))*dx(s + 1)
            return (b0,)
        elif term == "bt" or term == "bt_restricted":
            p = self.p
            if term == "bt":
                v = self.v
            elif term == "bt_restricted":
                v = self.r
            tensor_chi = self.tensor_chi
            bt0 = 0
            for s in range(4):
                bt0 += - p*tr(tensor_chi[s]*grad(v))*dx(s + 1)
            return (bt0,)
        elif term == "f":
            v = self.v
            det_deformation_gradient = self.det_deformation_gradient
            f0 = 0
            for s in range(4):
                f0 += inner(self.f, v)*det_deformation_gradient[s]*dx(s + 1)
            return (f0,)
        elif term == "g":
            q = self.q
            det_deformation_gradient = self.det_deformation_gradient
            g0 = 0
            for s in range(4):
                g0 += self.g*q*det_deformation_gradient[s]*dx(s + 1)
            return (g0,)
        elif term == "dirichlet_bc_u" or term == "dirichlet_bc_s":
            if term == "dirichlet_bc_u":
                V_s = self.V.sub(0)
            elif term == "dirichlet_bc_s":
                V_s = self.V.sub(0).collapse()
            bc0 = [DirichletBC(V_s, Constant((0.0, 0.0)), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product_u" or term == "inner_product_s":
            if term == "inner_product_u":
                u = self.u
                v = self.v
            elif term == "inner_product_s":
                u = self.s
                v = self.r
            x0 = inner(grad(u),grad(v))*dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(p, q)*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

@EIM(basis_generation="POD")
@ShapeParametrization(*shape_parametrization)
class AdvectionDiffusion(EllipticCoerciveProblem):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        assert "stokes_problem" in kwargs
        self.subdomains, self.boundaries, self.stokes_problem = kwargs["subdomains"], kwargs["boundaries"], kwargs["stokes_problem"]
        self.c = TrialFunction(V)
        self.d = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        #
        self.f = Constant(0.0)
        # Store parametrized tensors related to shape parametrization
        expression_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        scalar_element = V.ufl_element()
        tensor_element = TensorElement(scalar_element)
        det_deformation_gradient = (
            "mu[4]*mu[0]", # subdomain 1
            "mu[1]*mu[3]", # subdomain 2
            "mu[1]*mu[0]", # subdomain 3
            "mu[1]*mu[2]"  # subdomain 4
        )
        tensor_kappa = (
            (("(pow(mu[0], 2) + pow(tan(mu[5]), 2))/(mu[0]*mu[4])", "- tan(mu[5])/mu[0]"), ("- tan(mu[5])/mu[0]", "mu[4]/mu[0]")), # subdomain 1
            (("mu[3]/mu[1]", "0."), ("0.", "mu[1]/mu[3]")), # subdomain 2
            (("mu[0]/mu[1]", "0."), ("0.", "mu[1]/mu[0]")), # subdomain 3
            (("mu[2]/mu[1]", "0."), ("0.", "mu[1]/mu[2]"))  # subdomain 4
        )
        tensor_chi = (
            (("mu[0]", "-tan(mu[5])"), ("0.", "mu[4]")), # subdomain 1
            (("mu[3]", "0."), ("0.", "mu[1]")), # subdomain 2
            (("mu[0]", "0."), ("0.", "mu[1]")), # subdomain 3
            (("mu[2]", "0."), ("0.", "mu[1]"))  # subdomain 4
        )
        self.det_deformation_gradient = list()
        self.tensor_kappa = list()
        self.tensor_chi = list()
        for s in range(4):
            self.det_deformation_gradient.append(ParametrizedExpression(self, det_deformation_gradient[s], mu=expression_mu, element=scalar_element))
            self.tensor_kappa.append(ParametrizedExpression(self, tensor_kappa[s], mu=expression_mu, element=tensor_element))
            self.tensor_chi.append(ParametrizedExpression(self, tensor_chi[s], mu=expression_mu, element=tensor_element))
            
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
        
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "a":
            theta_a0 = 1.
            theta_a1 = 100.
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        elif term == "dirichlet_bc":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
                    
    ## Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        d = self.d
        dx = self.dx
        if term == "a":
            c = self.c
            (vel, _) = split(self.stokes_problem._solution)
            tensor_kappa = self.tensor_kappa
            tensor_chi = self.tensor_chi
            a0 = 0
            a1 = 0
            for s in range(4):
                a0 += inner(tensor_kappa[s]*grad(c), grad(d))*dx(s + 1)
                a1 += inner(tensor_chi[s]*vel, grad(c))*d*dx(s + 1)
            return (a0, a1)
        elif term == "f":
            det_deformation_gradient = self.det_deformation_gradient
            f0 = 0
            for s in range(4):
                f0 += self.f*d*det_deformation_gradient[s]*dx(s + 1)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(1.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 3)]
            return (bc0,)
        elif term == "inner_product":
            c = self.c
            x0 = inner(grad(c),grad(d))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 

#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 17: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/t_bypass.xml")
subdomains = MeshFunction("size_t", mesh, "data/t_bypass_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/t_bypass_facet_region.xml")

# 2a. Create Finite Element space for Stokes problem (Taylor-Hood P2-P1)
element_u  = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_p  = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element_up = MixedElement(element_u, element_p)
V = FunctionSpace(mesh, element_up, components=[["u", "s"], "p"])

# 3a. Allocate an object of the Stokes class
stokes_problem = Stokes(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [ \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0.5, 1.5), \
    (0., pi/6.) \
]
stokes_problem.set_mu_range(mu_range)

# 4a. Prepare reduction with a POD-Galerkin method
stokes_pod_galerkin_method = PODGalerkin(stokes_problem)
stokes_pod_galerkin_method.set_Nmax(25, EIM={"a": 3, "b": 3, "bt": 3, "bt_restricted": 3, "f": 1, "g": 1})

# 5a. Perform the offline phase
stokes_pod_galerkin_method.initialize_training_set(100, sampling=LinearlyDependentUniformDistribution(), EIM={"a": 4, "b": 4, "bt": 4, "bt_restricted": 4, "f": 2, "g": 2})
reduced_stokes_problem = stokes_pod_galerkin_method.offline()

# 6a. Perform an online solve
online_mu = (1.0, 1.0, 1.0, 1.0, 1.0, pi/6.)
reduced_stokes_problem.set_mu(online_mu)
reduced_stokes_problem.solve()
reduced_stokes_problem.export_solution("Stokes", "online_solution")

# 2b. Create Finite Element space for advection diffusion problem
element_c = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
C = FunctionSpace(mesh, element_c)

# 3b. Allocate an object of the AdvectionDiffusionProblem class
advection_diffusion_problem = AdvectionDiffusion(C, subdomains=subdomains, boundaries=boundaries, stokes_problem=stokes_problem)
advection_diffusion_problem.set_mu_range(mu_range)

# 4b. Prepare reduction with a POD-Galerkin method
advection_diffusion_pod_galerkin_method = PODGalerkin(advection_diffusion_problem)
advection_diffusion_pod_galerkin_method.set_Nmax(25, EIM={"a": (3, 25), "f": 1})

# 5b. Perform the offline phase
first_mu = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
advection_diffusion_problem.set_mu(first_mu)
advection_diffusion_pod_galerkin_method.initialize_training_set(100, sampling=LinearlyDependentUniformDistribution(), EIM={"a": (4, 100), "f": 2})
reduced_advection_diffusion_problem = advection_diffusion_pod_galerkin_method.offline()

# 6b. Perform an online solve
reduced_advection_diffusion_problem.set_mu(online_mu)
reduced_advection_diffusion_problem.solve()
reduced_advection_diffusion_problem.export_solution("AdvectionDiffusion", "online_solution")

# 7a. Perform an error analysis
stokes_pod_galerkin_method.initialize_testing_set(100, sampling=LinearlyDependentUniformDistribution(), EIM=10)
#stokes_pod_galerkin_method.error_analysis()

# 7b. Perform an error analysis
advection_diffusion_pod_galerkin_method.initialize_testing_set(100, sampling=LinearlyDependentUniformDistribution(), EIM=100)
advection_diffusion_pod_galerkin_method.error_analysis()

# 8a. Perform a speedup analysis
stokes_pod_galerkin_method.speedup_analysis()

# 8b. Perform a speedup analysis
advection_diffusion_pod_galerkin_method.speedup_analysis()
