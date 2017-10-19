# Copyright (C) 2015- 2017 by the RBniCS authors
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

import pytest
import os
import itertools
import functools
from contextlib import contextmanager
from dolfin import CellSize, Constant, cos, div, Expression, FiniteElement, FunctionSpace, grad, inner, Measure, Mesh, MeshFunction, MixedElement, pi, sin, split, sqrt, tan, TestFunction, TrialFunction, VectorElement
from rbnics import ShapeParametrization
from rbnics.backends.dolfin.wrapping import PullBackFormsToReferenceDomain
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import forms_are_close
from rbnics.eim.problems import EIM
from rbnics.problems.base import ParametrizedProblem

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "test_pull_back_to_reference_domain")

def theta_times_operator(problem, term):
    return sum([Constant(theta)*operator for (theta, operator) in zip(problem.compute_theta(term), problem.assemble_operator(term))])

def keep_shape_parametrization_affine(shape_parametrization_expression):
    return shape_parametrization_expression

def make_shape_parametrization_non_affine(shape_parametrization_expression):
    non_affine_shape_parametrization_expression = list()
    for shape_parametrization_expression_on_subdomain in shape_parametrization_expression:
        non_affine_shape_parametrization_expression_on_subdomain = list()
        for (coord, expression_coord) in enumerate(shape_parametrization_expression_on_subdomain):
            non_affine_shape_parametrization_expression_on_subdomain.append(
                expression_coord + " + 1e-16*mu[0]*x[" + str(coord) + "]**2"
            )
        non_affine_shape_parametrization_expression.append(tuple(non_affine_shape_parametrization_expression_on_subdomain))
    return non_affine_shape_parametrization_expression
    
def NoEIM():
    def NoEIM_decorator(Class):
        return Class
    return NoEIM_decorator
    
def raises(ExceptionType):
    """
        Wrapper around pytest.raises to support None.
        Credits: https://github.com/pytest-dev/pytest/issues/1830
    """
    if ExceptionType is not None:
        return pytest.raises(ExceptionType)
    else:
        @contextmanager
        def not_raises():
            try:
                yield
            except Exception as e:
                raise e
        return not_raises()
    
def check_affine_and_non_affine_shape_parametrizations(*decorator_args):
    decorators = list()
    decorators.append(
        pytest.mark.parametrize("shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message", [
            (keep_shape_parametrization_affine, NoEIM, None, None),
            (make_shape_parametrization_non_affine, NoEIM, AssertionError, "Non affine parametric dependence detected. Please use one among DEIM, EIM and ExactParametrizedFunctions"),
            (make_shape_parametrization_non_affine, EIM, None, None)
        ])
    )
    for decorator_arg in decorator_args:
        decorators.append(pytest.mark.parametrize(decorator_arg[0], decorator_arg[1]))
    
    def check_affine_and_non_affine_shape_parametrizations_decorator(original_test):
        @functools.wraps(original_test)
        def test_with_exception_check(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message, **kwargs):
            with raises(ExceptionType) as excinfo:
                original_test(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message, **kwargs)
            if ExceptionType is not None:
                assert str(excinfo.value) == exception_message
        decorated_test = test_with_exception_check
        for decorator in decorators:
            decorated_test = decorator(decorated_test)
        return decorated_test
        
    return check_affine_and_non_affine_shape_parametrizations_decorator

# Test forms pull back to reference domain for tutorial 3
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_hole(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "hole.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("2 - 2*mu[0] + mu[0]*x[0] + (2 - 2*mu[0])*x[1]", "2 - 2*mu[1] + (2 - mu[1])*x[1]"), # subdomain 1
        ("2*mu[0]- 2 + x[0] + (mu[0] - 1)*x[1]", "2 - 2*mu[1] + (2 - mu[1])*x[1]"), # subdomain 2
        ("2 - 2*mu[0] + (2 - mu[0])*x[0]", "2 - 2*mu[1] + (2- 2*mu[1])*x[0] + mu[1]*x[1]"), # subdomain 3
        ("2 - 2*mu[0] + (2 - mu[0])*x[0]", "2*mu[1] - 2 + (mu[1] - 1)*x[0] + x[1]"), # subdomain 4
        ("2*mu[0] - 2 + (2 - mu[0])*x[0]", "2 - 2*mu[1] + (2*mu[1]- 2)*x[0] + mu[1]*x[1]"), # subdomain 5
        ("2*mu[0] - 2 + (2 - mu[0])*x[0]", "2*mu[1] - 2 + (1 - mu[1])*x[0] + x[1]"), # subdomain 6
        ("2 - 2*mu[0] + mu[0]*x[0] + (2*mu[0] - 2)*x[1]", "2*mu[1] - 2 + (2 - mu[1])*x[1]"), # subdomain 7
        ("2*mu[0] - 2 + x[0] + (1 - mu[0])*x[1]", "2*mu[1] - 2 + (2 - mu[1])*x[1]") # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)
    
    # Define base problem
    class Hole(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1., 0)
            self.mu_range = [(0.5, 1.5), (0.5, 1.5), (0.01, 1.0)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleOnReferenceDomain(Hole):
        def __init__(self, V, **kwargs):
            Hole.__init__(self, "HoleOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            m1 = self.mu[0]
            m2 = self.mu[1]
            m3 = self.mu[2]
            if term == "a":
                # subdomains 1 and 7
                theta_a0 = - (m2 - 2)/m1 - (2*(2*m1 - 2)*(m1 - 1))/(m1*(m2 - 2)) # K11
                theta_a1 = -m1/(m2 - 2) # K22
                theta_a2 = -(2*(m1 - 1))/(m2 - 2) # K12 and K21
                # subdomains 2 and 8
                theta_a3 = 2 - (m1 - 1)*(m1 - 1)/(m2 - 2) - m2
                theta_a4 = -1/(m2 - 2)
                theta_a5 = (m1 - 1)/(m2 - 2)
                # subdomains 3 and 5
                theta_a6 = -m2/(m1 - 2)
                theta_a7 = - (m1 - 2)/m2 - (2*(2*m2 - 2)*(m2 - 1))/(m2*(m1 - 2))
                theta_a8 = -(2*(m2 - 1))/(m1 - 2)
                # subdomains 4 and 6
                theta_a9 = -1/(m1 - 2)
                theta_a10 = 2 - (m2 - 1)*(m2 - 1)/(m1 - 2) - m1
                theta_a11 = (m2 - 1)/(m1 - 2)
                # boundaries 5, 6, 7 and 8
                theta_a12 = m3
                # Return
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8, theta_a9, theta_a10, theta_a11, theta_a12)
            elif term == "f":
                theta_f0 = m1 # boundary 1
                theta_f1 = m2 # boundary 2
                theta_f2 = m1 # boundary 3
                theta_f3 = m2 # boundary 4
                # Return
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                # subdomains 1 and 7
                a0 = inner(u.dx(0), v.dx(0))*dx(1) + inner(u.dx(0), v.dx(0))*dx(7)
                a1 = inner(u.dx(1), v.dx(1))*dx(1) + inner(u.dx(1), v.dx(1))*dx(7)
                a2 = inner(u.dx(0), v.dx(1))*dx(1) + inner(u.dx(1), v.dx(0))*dx(1) - (inner(u.dx(0), v.dx(1))*dx(7) + inner(u.dx(1), v.dx(0))*dx(7))
                # subdomains 2 and 8
                a3 = inner(u.dx(0), v.dx(0))*dx(2) + inner(u.dx(0), v.dx(0))*dx(8)
                a4 = inner(u.dx(1), v.dx(1))*dx(2) + inner(u.dx(1), v.dx(1))*dx(8)
                a5 = inner(u.dx(0), v.dx(1))*dx(2) + inner(u.dx(1), v.dx(0))*dx(2) - (inner(u.dx(0), v.dx(1))*dx(8) + inner(u.dx(1), v.dx(0))*dx(8))
                # subdomains 3 and 5
                a6 = inner(u.dx(0), v.dx(0))*dx(3) + inner(u.dx(0), v.dx(0))*dx(5)
                a7 = inner(u.dx(1), v.dx(1))*dx(3) + inner(u.dx(1), v.dx(1))*dx(5)
                a8 = inner(u.dx(0), v.dx(1))*dx(3) + inner(u.dx(1), v.dx(0))*dx(3) - (inner(u.dx(0), v.dx(1))*dx(5) + inner(u.dx(1), v.dx(0))*dx(5))
                # subdomains 4 and 6
                a9 = inner(u.dx(0), v.dx(0))*dx(4) + inner(u.dx(0), v.dx(0))*dx(6)
                a10 = inner(u.dx(1), v.dx(1))*dx(4) + inner(u.dx(1), v.dx(1))*dx(6)
                a11 = inner(u.dx(0), v.dx(1))*dx(4) + inner(u.dx(1), v.dx(0))*dx(4) - (inner(u.dx(0), v.dx(1))*dx(6) + inner(u.dx(1), v.dx(0))*dx(6))
                # boundaries 5, 6, 7 and 8
                a12 = inner(u, v)*ds(5) + inner(u, v)*ds(6) + inner(u, v)*ds(7) + inner(u, v)*ds(8)
                # Return
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
            elif term == "f":
                f0 = v*ds(1) # boundary 1
                f1 = v*ds(2) # boundary 2
                f2 = v*ds(3) # boundary 3
                f3 = v*ds(4) # boundary 4
                # Return
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "f", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class HolePullBack(Hole):
        def __init__(self, V, **kwargs):
            Hole.__init__(self, "HolePullBack")
            self.V = V
            
        def compute_theta(self, term):
            m3 = self.mu[2]
            if term == "a":
                theta_a0 = 1.0
                theta_a1 = m3
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx
                a1 = inner(u, v)*ds(5) + inner(u, v)*ds(6) + inner(u, v)*ds(7) + inner(u, v)*ds(8)
                return (a0, a1)
            elif term == "f":
                f0 = v*ds(1) + v*ds(2) + v*ds(3) + v*ds(4)
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Check forms
    problem_on_reference_domain = HoleOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = HolePullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
# Test forms pull back to reference domain for tutorial 3 rotation
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_hole_rotation(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "hole.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "hole_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("-2*sqrt(2.0)*cos(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) + x[1]*(-sqrt(2.0)*sin(mu[0])/2 - 3*sqrt(2.0)*cos(mu[0])/2 + 2) + 2", "-2*sqrt(2.0)*sin(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 - sqrt(2.0)*cos(mu[0])/2) + x[1]*(-3*sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2 + 2) + 2"), # subdomain 1
        ("2*sqrt(2.0)*sin(mu[0]) + x[0] + x[1]*(sqrt(2.0)*sin(mu[0]) - 1) - 2", "-2*sqrt(2.0)*cos(mu[0]) + x[1]*(-sqrt(2.0)*cos(mu[0]) + 2) + 2"), # subdomain 2
        ("-2*sqrt(2.0)*cos(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 - 3*sqrt(2.0)*cos(mu[0])/2 + 2) + x[1]*(-sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) + 2", "-2*sqrt(2.0)*sin(mu[0]) + x[0]*(-3*sqrt(2.0)*sin(mu[0])/2 - sqrt(2.0)*cos(mu[0])/2 + 2) + x[1]*(sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) + 2"), # subdomain 3
        ("-2*sqrt(2.0)*sin(mu[0]) + x[0]*(-sqrt(2.0)*sin(mu[0]) + 2) + 2", "2*sqrt(2.0)*cos(mu[0]) + x[0]*(sqrt(2.0)*cos(mu[0]) - 1) + x[1] - 2"), # subdomain 4
        ("2*sqrt(2.0)*sin(mu[0]) + x[0]*(-3*sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2 + 2) + x[1]*(-sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) - 2", "-2*sqrt(2.0)*cos(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 + 3*sqrt(2.0)*cos(mu[0])/2 - 2) + x[1]*(sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) + 2"), # subdomain 5
        ("2*sqrt(2.0)*cos(mu[0]) + x[0]*(-sqrt(2.0)*cos(mu[0]) + 2) - 2", "2*sqrt(2.0)*sin(mu[0]) + x[0]*(-sqrt(2.0)*sin(mu[0]) + 1) + x[1] - 2"), # subdomain 6
        ("-2*sqrt(2.0)*sin(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2) + x[1]*(3*sqrt(2.0)*sin(mu[0])/2 + sqrt(2.0)*cos(mu[0])/2 - 2) + 2", "2*sqrt(2.0)*cos(mu[0]) + x[0]*(sqrt(2.0)*sin(mu[0])/2 - sqrt(2.0)*cos(mu[0])/2) + x[1]*(sqrt(2.0)*sin(mu[0])/2 - 3*sqrt(2.0)*cos(mu[0])/2 + 2) - 2"), # subdomain 7
        ("2*sqrt(2.0)*cos(mu[0]) + x[0] + x[1]*(-sqrt(2.0)*cos(mu[0]) + 1) - 2", "2*sqrt(2.0)*sin(mu[0]) + x[1]*(-sqrt(2.0)*sin(mu[0]) + 2) - 2")  # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)
    
    # Define base problem
    class HoleRotation(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (pi/4.0, 0.01)
            self.mu_range = [(pi/4.0-pi/45.0, pi/4.0+pi/45.0), (0.01, 1.0)]
            
        def init(self):
            pass
    
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleRotationOnReferenceDomain(HoleRotation):
        def __init__(self, V, **kwargs):
            HoleRotation.__init__(self, "HoleRotationOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu = self.mu
            mu1 = mu[0]
            mu2 = mu[1]
            if term == "a":
                theta_a0 = (-5*sqrt(2.0)**2 + 16*sqrt(2.0)*sin(mu1) + 8*sqrt(2.0)*cos(mu1) - 16)/(sqrt(2.0)*(sqrt(2.0) - 4*cos(mu1)))
                theta_a1 = -sqrt(2.0)/(sqrt(2.0) - 4*cos(mu1))
                theta_a2 = (-2*sqrt(2.0) + 4*sin(mu1))/(sqrt(2.0) - 4*cos(mu1))
                theta_a3 = (-sqrt(2.0)**2 + 2*sqrt(2.0)*sin(mu1) + 4*sqrt(2.0)*cos(mu1) - 5)/(sqrt(2.0)*cos(mu1) - 2)
                theta_a4 = -1/(sqrt(2.0)*cos(mu1) - 2)
                theta_a5 = (sqrt(2.0)*sin(mu1) - 1)/(sqrt(2.0)*cos(mu1) - 2)
                theta_a6 = -sqrt(2.0)/(sqrt(2.0) - 4*sin(mu1))
                theta_a7 = (-5*sqrt(2.0)**2 + 8*sqrt(2.0)*sin(mu1) + 16*sqrt(2.0)*cos(mu1) - 16)/(sqrt(2.0)*(sqrt(2.0) - 4*sin(mu1)))
                theta_a8 = (-2*sqrt(2.0) + 4*cos(mu1))/(sqrt(2.0) - 4*sin(mu1))
                theta_a9 = -1/(sqrt(2.0)*sin(mu1) - 2)
                theta_a10 = (-sqrt(2.0)**2 + 4*sqrt(2.0)*sin(mu1) + 2*sqrt(2.0)*cos(mu1) - 5)/(sqrt(2.0)*sin(mu1) - 2)
                theta_a11 = (sqrt(2.0)*cos(mu1) - 1)/(sqrt(2.0)*sin(mu1) - 2)
                theta_a12 = -sqrt(2.0)/(sqrt(2.0) - 4*cos(mu1))
                theta_a13 = (-5*sqrt(2.0)**2 + 16*sqrt(2.0)*sin(mu1) + 8*sqrt(2.0)*cos(mu1) - 16)/(sqrt(2.0)*(sqrt(2.0) - 4*cos(mu1)))
                theta_a14 = 2*(sqrt(2.0) - 2*sin(mu1))/(sqrt(2.0) - 4*cos(mu1))
                theta_a15 = -1/(sqrt(2.0)*cos(mu1) - 2)
                theta_a16 = (-sqrt(2.0)**2 + 2*sqrt(2.0)*sin(mu1) + 4*sqrt(2.0)*cos(mu1) - 5)/(sqrt(2.0)*cos(mu1) - 2)
                theta_a17 = (-sqrt(2.0)*sin(mu1) + 1)/(sqrt(2.0)*cos(mu1) - 2)
                theta_a18 = (-5*sqrt(2.0)**2 + 8*sqrt(2.0)*sin(mu1) + 16*sqrt(2.0)*cos(mu1) - 16)/(sqrt(2.0)*(sqrt(2.0) - 4*sin(mu1)))
                theta_a19 = -sqrt(2.0)/(sqrt(2.0) - 4*sin(mu1))
                theta_a20 = 2*(sqrt(2.0) - 2*cos(mu1))/(sqrt(2.0) - 4*sin(mu1))
                theta_a21 = (-sqrt(2.0)**2 + 4*sqrt(2.0)*sin(mu1) + 2*sqrt(2.0)*cos(mu1) - 5)/(sqrt(2.0)*sin(mu1) - 2)
                theta_a22 = -1/(sqrt(2.0)*sin(mu1) - 2)
                theta_a23 = (-sqrt(2.0)*cos(mu1) + 1)/(sqrt(2.0)*sin(mu1) - 2)
                theta_a24 = mu2
                theta_a25 = mu2
                theta_a26 = mu2
                theta_a27 = mu2
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8, theta_a9, theta_a10, theta_a11, theta_a12, theta_a13, theta_a14, theta_a15, theta_a16, theta_a17, theta_a18, theta_a19, theta_a20, theta_a21, theta_a22, theta_a23, theta_a24, theta_a25, theta_a26, theta_a27)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = 1.0
                theta_f2 = 1.0
                theta_f3 = 1.0
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")
        
        def assemble_operator(self, term):
            if term == "a":
                a0 = u.dx(0)*v.dx(0)*dx(1)
                a1 = u.dx(1)*v.dx(1)*dx(1)
                a2 = u.dx(0)*v.dx(1)*dx(1) + u.dx(1)*v.dx(0)*dx(1)
                a3 = u.dx(0)*v.dx(0)*dx(2)
                a4 = u.dx(1)*v.dx(1)*dx(2)
                a5 = u.dx(0)*v.dx(1)*dx(2) + u.dx(1)*v.dx(0)*dx(2)
                a6 = u.dx(0)*v.dx(0)*dx(3)
                a7 = u.dx(1)*v.dx(1)*dx(3)
                a8 = u.dx(0)*v.dx(1)*dx(3) + u.dx(1)*v.dx(0)*dx(3)
                a9 = u.dx(0)*v.dx(0)*dx(4)
                a10 = u.dx(1)*v.dx(1)*dx(4)
                a11 = u.dx(0)*v.dx(1)*dx(4) + u.dx(1)*v.dx(0)*dx(4)
                a12 = u.dx(0)*v.dx(0)*dx(5)
                a13 = u.dx(1)*v.dx(1)*dx(5)
                a14 = u.dx(0)*v.dx(1)*dx(5) + u.dx(1)*v.dx(0)*dx(5)
                a15 = u.dx(0)*v.dx(0)*dx(6)
                a16 = u.dx(1)*v.dx(1)*dx(6)
                a17 = u.dx(0)*v.dx(1)*dx(6) + u.dx(1)*v.dx(0)*dx(6)
                a18 = u.dx(0)*v.dx(0)*dx(7)
                a19 = u.dx(1)*v.dx(1)*dx(7)
                a20 = u.dx(0)*v.dx(1)*dx(7) + u.dx(1)*v.dx(0)*dx(7)
                a21 = u.dx(0)*v.dx(0)*dx(8)
                a22 = u.dx(1)*v.dx(1)*dx(8)
                a23 = u.dx(0)*v.dx(1)*dx(8) + u.dx(1)*v.dx(0)*dx(8)
                a24 = u*v*ds(5)
                a25 = u*v*ds(6)
                a26 = u*v*ds(7)
                a27 = u*v*ds(8)
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27)
            elif term == "f":
                f0 = v*ds(1)
                f1 = v*ds(2)
                f2 = v*ds(3)
                f3 = v*ds(4)
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")
            
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "f", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class HoleRotationPullBack(HoleRotation):
        def __init__(self, V, **kwargs):
            HoleRotation.__init__(self, "HoleRotationPullBack")
            self.V = V
            
        def compute_theta(self, term):
            mu = self.mu
            mu2 = mu[1]
            if term == "a":
                theta_a0 = 1.0
                theta_a1 = mu2
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0,)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx
                a1 = u*v*ds(5) + u*v*ds(6) + u*v*ds(7) + u*v*ds(8)
                return (a0, a1)
            elif term == "f":
                f0 = v*ds(1) + v*ds(2) + v*ds(3) + v*ds(4)
                return (f0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Check forms
    problem_on_reference_domain = HoleRotationOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = HoleRotationPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)

# Test forms pull back to reference domain for tutorial 4
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_graetz(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "graetz.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"), # subdomain 1
        ("mu[0]*(x[0] - 1) + 1", "x[1]") # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    vel = Expression("x[1]*(1-x[1])", element=V.ufl_element())
    ff = Constant(1.)
    
    # Define base problem
    class Graetz(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(0.1, 10.0), (0.01, 10.0)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzOnReferenceDomain(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu1 = self.mu[0]
            mu2 = self.mu[1]
            if term == "a":
                theta_a0 = mu2
                theta_a1 = mu2/mu1
                theta_a2 = mu1*mu2
                theta_a3 = 1.0
                return (theta_a0, theta_a1, theta_a2, theta_a3)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu1
                return (theta_f0, theta_f1)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx(1)
                a1 = u.dx(0)*v.dx(0)*dx(2)
                a2 = u.dx(1)*v.dx(1)*dx(2)
                a3 = vel*u.dx(0)*v*dx(1) + vel*u.dx(0)*v*dx(2)
                return (a0, a1, a2, a3)
            elif term == "f":
                f0 = ff*v*dx(1)
                f1 = ff*v*dx(2)
                return (f0, f1)
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "f", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzPullBack(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzPullBack")
            self.V = V
            
        def compute_theta(self, term):
            mu2 = self.mu[1]
            if term == "a":
                theta_a0 = mu2
                theta_a1 = 1.0
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx
                a1 = vel*u.dx(0)*v*dx
                return (a0, a1)
            elif term == "f":
                f0 = ff*v*dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Check forms
    problem_on_reference_domain = GraetzOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = GraetzPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
# Test forms pull back to reference domain for tutorial 9
@check_affine_and_non_affine_shape_parametrizations((
    "CellSize, cell_size_pull_back", [
        (lambda mesh: Constant(0.), lambda mu1: 0),
        (lambda mesh: Constant(1.), lambda mu1: 1),
        (CellSize, lambda mu1: sqrt(mu1))
    ]
))
def test_pull_back_to_reference_domain_advection_dominated(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message, CellSize, cell_size_pull_back):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "graetz.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "graetz_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"), # subdomain 1
        ("mu[0]*(x[0] - 1) + 1", "x[1]") # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(V)
    v = TestFunction(V)
    dx = Measure("dx")(subdomain_data=subdomains)
    vel = Expression("x[1]*(1-x[1])", element=V.ufl_element())
    ff = Constant(1.)
    h = CellSize(V.mesh())
    
    # Define base problem
    class Graetz(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(0.5, 4.0), (1e-6, 1e-1)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzOnReferenceDomain(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu1 = self.mu[0]
            mu2 = self.mu[1]
            if term == "a":
                theta_a0 = mu2
                theta_a1 = mu2/mu1
                theta_a2 = mu2*mu1
                theta_a3 = 1.0
                theta_a4 = mu2
                theta_a5 = mu2/mu1*cell_size_pull_back(mu1)
                theta_a6 = mu2*mu1*cell_size_pull_back(mu1)
                theta_a7 = 1.0
                theta_a8 = cell_size_pull_back(mu1)
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu1
                theta_f2 = 1.0
                theta_f3 = mu1*cell_size_pull_back(mu1)
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx(1)
                a1 = u.dx(0)*v.dx(0)*dx(2)
                a2 = u.dx(1)*v.dx(1)*dx(2)
                a3 = vel*u.dx(0)*v*dx(1) + vel*u.dx(0)*v*dx(2)
                a4 = - h*inner(div(grad(u)), v)*dx(1)
                a5 = - h*u.dx(0).dx(0)*v*dx(2)
                a6 = - h*u.dx(1).dx(1)*v*dx(2)
                a7 = h*vel*u.dx(0)*v*dx(1)
                a8 = h*vel*u.dx(0)*v*dx(2)
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
            elif term == "f":
                f0 = ff*v*dx(1)
                f1 = ff*v*dx(2)
                f2 = h*ff*v*dx(1)
                f3 = h*ff*v*dx(2)
                return (f0, f1, f2, f3)
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "f", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class GraetzPullBack(Graetz):
        def __init__(self, V, **kwargs):
            Graetz.__init__(self, "GraetzPullBack")
            self.V = V
            
        def compute_theta(self, term):
            mu2 = self.mu[1]
            if term == "a":
                theta_a0 = mu2
                theta_a1 = 1.0
                return (theta_a0, theta_a1)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx - h*inner(div(grad(u)), v)*dx
                a1 = vel*u.dx(0)*v*dx + h*vel*u.dx(0)*v*dx
                return (a0, a1)
            elif term == "f":
                f0 = ff*v*dx + h*ff*v*dx
                return (f0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Check forms
    problem_on_reference_domain = GraetzOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = GraetzPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
# Test forms pull back to reference domain for tutorial 17
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "t_bypass.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "t_bypass_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[4]*x[0] + mu[1] - mu[4]", "mu[4]*tan(mu[5])*x[0] + mu[0]*x[1] + mu[2] - mu[4]*tan(mu[5]) - mu[0]"), # subdomain 1
        ("mu[4]*x[0] + mu[1] - mu[4]", "mu[4]*tan(mu[5])*x[0] + mu[0]*x[1] + mu[2] - mu[4]*tan(mu[5]) - mu[0]"), # subdomain 2
        ("mu[1]*x[0]", "mu[3]*x[1] + mu[2] + mu[0] - 2*mu[3]"), # subdomain 3
        ("mu[1]*x[0]", "mu[3]*x[1] + mu[2] + mu[0] - 2*mu[3]"), # subdomain 4
        ("mu[1]*x[0]", "mu[0]*x[1] + mu[2] - mu[0]"), # subdomain 5
        ("mu[1]*x[0]", "mu[0]*x[1] + mu[2] - mu[0]"), # subdomain 6
        ("mu[1]*x[0]", "mu[2]*x[1]"), # subdomain 7
        ("mu[1]*x[0]", "mu[2]*x[1]"), # subdomain 8
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    up = TrialFunction(V)
    (u, p) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)
    
    ff = Constant((0.0, -10.0))
    gg = Constant(1.0)
    nu = 1.0
    
    # Define base problem
    class Stokes(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1., 1., 1., 1., 0.)
            self.mu_range = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.0, pi/6.0)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOnReferenceDomain(Stokes):
        def __init__(self, V, **kwargs):
            Stokes.__init__(self, "StokesOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu = self.mu
            mu1 = mu[0]
            mu2 = mu[1]
            mu3 = mu[2]
            mu4 = mu[3]
            mu5 = mu[4]
            mu6 = mu[5]
            if term == "a":
                theta_a0 = nu*(mu1/mu5)
                theta_a1 = nu*(-tan(mu6))
                theta_a2 = nu*(mu5*(tan(mu6)**2 + 1)/mu1)
                theta_a3 = nu*(mu4/mu2)
                theta_a4 = nu*(mu2/mu4)
                theta_a5 = nu*(mu1/mu2)
                theta_a6 = nu*(mu2/mu1)
                theta_a7 = nu*(mu3/mu2)
                theta_a8 = nu*(mu2/mu3)
                return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5, theta_a6, theta_a7, theta_a8)
            elif term in ("b", "bt"):
                theta_b0 = mu1
                theta_b1 = -tan(mu6)*mu5
                theta_b2 = mu5
                theta_b3 = mu4
                theta_b4 = mu2
                theta_b5 = mu1
                theta_b6 = mu2
                theta_b7 = mu3
                theta_b8 = mu2
                return (theta_b0, theta_b1, theta_b2, theta_b3, theta_b4, theta_b5, theta_b6, theta_b7, theta_b8)
            elif term == "f":
                theta_f0 = mu[0]*mu[4]
                theta_f1 = mu[1]*mu[3]
                theta_f2 = mu[0]*mu[1]
                theta_f3 = mu[1]*mu[2]
                return (theta_f0, theta_f1, theta_f2, theta_f3)
            elif term == "g":
                theta_g0 = mu[0]*mu[4]
                theta_g1 = mu[1]*mu[3]
                theta_g2 = mu[0]*mu[1]
                theta_g3 = mu[1]*mu[2]
                return (theta_g0, theta_g1, theta_g2, theta_g3)
            else:
                raise ValueError("Invalid term for compute_theta().")
            
        def assemble_operator(self, term):
            if term == "a":
                a0 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*(dx(1) + dx(2))
                a1 = (u[0].dx(0)*v[0].dx(1) + u[0].dx(1)*v[0].dx(0) + u[1].dx(0)*v[1].dx(1) + u[1].dx(1)*v[1].dx(0))*(dx(1) + dx(2))
                a2 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*(dx(1) + dx(2))
                a3 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*(dx(3) + dx(4))
                a4 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*(dx(3) + dx(4))
                a5 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*(dx(5) + dx(6))
                a6 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*(dx(5) + dx(6))
                a7 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*(dx(7) + dx(8))
                a8 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*(dx(7) + dx(8))
                return (a0, a1, a2, a3, a4, a5, a6, a7, a8)
            elif term == "b":
                b0 = - q*u[0].dx(0)*(dx(1) + dx(2))
                b1 = - q*u[0].dx(1)*(dx(1) + dx(2))
                b2 = - q*u[1].dx(1)*(dx(1) + dx(2))
                b3 = - q*u[0].dx(0)*(dx(3) + dx(4))
                b4 = - q*u[1].dx(1)*(dx(3) + dx(4))
                b5 = - q*u[0].dx(0)*(dx(5) + dx(6))
                b6 = - q*u[1].dx(1)*(dx(5) + dx(6))
                b7 = - q*u[0].dx(0)*(dx(7) + dx(8))
                b8 = - q*u[1].dx(1)*(dx(7) + dx(8))
                return (b0, b1, b2, b3, b4, b5, b6, b7, b8)
            elif term == "bt":
                bt0 = - p*v[0].dx(0)*(dx(1) + dx(2))
                bt1 = - p*v[0].dx(1)*(dx(1) + dx(2))
                bt2 = - p*v[1].dx(1)*(dx(1) + dx(2))
                bt3 = - p*v[0].dx(0)*(dx(3) + dx(4))
                bt4 = - p*v[1].dx(1)*(dx(3) + dx(4))
                bt5 = - p*v[0].dx(0)*(dx(5) + dx(6))
                bt6 = - p*v[1].dx(1)*(dx(5) + dx(6))
                bt7 = - p*v[0].dx(0)*(dx(7) + dx(8))
                bt8 = - p*v[1].dx(1)*(dx(7) + dx(8))
                return (bt0, bt1, bt2, bt3, bt4, bt5, bt6, bt7, bt8)
            elif term == "f":
                f0 = inner(ff, v)*(dx(1) + dx(2))
                f1 = inner(ff, v)*(dx(3) + dx(4))
                f2 = inner(ff, v)*(dx(5) + dx(6))
                f3 = inner(ff, v)*(dx(7) + dx(8))
                return (f0, f1, f2, f3)
            elif term == "g":
                g0 = gg*q*(dx(1) + dx(2))
                g1 = gg*q*(dx(3) + dx(4))
                g2 = gg*q*(dx(5) + dx(6))
                g3 = gg*q*(dx(7) + dx(8))
                return (g0, g1, g2, g3)
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "b", "bt", "f", "g", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesPullBack(Stokes):
        def __init__(self, V, **kwargs):
            Stokes.__init__(self, "StokesPullBack")
            self.V = V
            
        def compute_theta(self, term):
            if term == "a":
                theta_a0 = nu*1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx
                return (a0, )
            elif term == "b":
                b0 = - q*div(u)*dx
                return (b0, )
            elif term == "bt":
                bt0 = - p*div(v)*dx
                return (bt0, )
            elif term == "f":
                f0 = inner(ff, v)*dx
                return (f0, )
            elif term == "g":
                g0 = q*gg*dx
                return (g0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Check forms
    problem_on_reference_domain = StokesOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)
        
        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)
        
# Test forms pull back to reference domain for tutorial 18
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_elliptic_optimal_control_1(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "elliptic_optimal_control_1.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "elliptic_optimal_control_1_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "elliptic_optimal_control_1_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "x[1]"), # subdomain 1
        ("mu[0]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(scalar_element, scalar_element, scalar_element)
    V = FunctionSpace(mesh, element, components=["y", "u", "p"])
    yup = TrialFunction(V)
    (y, u, p) = split(yup)
    zvq = TestFunction(V)
    (z, v, q) = split(zvq)
    dx = Measure("dx")(subdomain_data=subdomains)
    alpha = Constant(0.01)
    y_d = Constant(1.0)
    ff = Constant(2.0)
    
    # Define base problem
    class EllipticOptimalControl(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., 1.)
            self.mu_range = [(1.0, 3.5), (0.5, 2.5)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class EllipticOptimalControlOnReferenceDomain(EllipticOptimalControl):
        def __init__(self, V, **kwargs):
            EllipticOptimalControl.__init__(self, "EllipticOptimalControlOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu1 = self.mu[0]
            mu2 = self.mu[1]
            if term in ("a", "a*"):
                theta_a0 = 1.0
                theta_a1 = 1.0/mu1
                theta_a2 = mu1
                return (theta_a0, theta_a1, theta_a2)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                theta_c1 = mu1
                return (theta_c0, theta_c1)
            elif term == "m":
                theta_m0 = 1.0
                theta_m1 = mu1
                return (theta_m0, theta_m1)
            elif term == "n":
                theta_n0 = alpha
                theta_n1 = alpha*mu1
                return (theta_n0, theta_n1)
            elif term == "f":
                theta_f0 = 1.0
                theta_f1 = mu1
                return (theta_f0, theta_f1)
            elif term == "g":
                theta_g0 = 1.0
                theta_g1 = mu1*mu2
                return (theta_g0, theta_g1)
            elif term == "h":
                theta_h0 = 1.0 + mu1*mu2**2
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(y), grad(q))*dx(1)
                a1 = y.dx(0)*q.dx(0)*dx(2)
                a2 = y.dx(1)*q.dx(1)*dx(2)
                return (a0, a1, a2)
            elif term == "a*":
                as0 = inner(grad(z), grad(p))*dx(1)
                as1 = z.dx(0)*p.dx(0)*dx(2)
                as2 = z.dx(1)*p.dx(1)*dx(2)
                return (as0, as1, as2)
            elif term == "c":
                c0 = u*q*dx(1)
                c1 = u*q*dx(2)
                return (c0, c1)
            elif term == "c*":
                cs0 = v*p*dx(1)
                cs1 = v*p*dx(2)
                return (cs0, cs1)
            elif term == "m":
                m0 = y*z*dx(1)
                m1 = y*z*dx(2)
                return (m0, m1)
            elif term == "n":
                n0 = u*v*dx(1)
                n1 = u*v*dx(2)
                return (n0, n1)
            elif term == "f":
                f0 = ff*q*dx(1)
                f1 = ff*q*dx(2)
                return (f0, f1)
            elif term == "g":
                g0 = y_d*z*dx(1)
                g1 = y_d*z*dx(2)
                return (g0, g1)
            elif term == "h":
                h0 = 1.0
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "a*", "c", "c*", "m", "n", "f", "g", "h", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class EllipticOptimalControlPullBack(EllipticOptimalControl):
        def __init__(self, V, **kwargs):
            EllipticOptimalControl.__init__(self, "EllipticOptimalControlPullBack")
            self.V = V
            
        def compute_theta(self, term):
            mu2 = self.mu[1]
            if term in ("a", "a*"):
                theta_a0 = 1.0
                return (theta_a0,)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                return (theta_c0,)
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha
                return (theta_n0,)
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0,)
            elif term == "g":
                theta_g0 = 1.0
                theta_g1 = mu2
                return (theta_g0, theta_g1)
            elif term == "h":
                theta_h0 = 1.0
                theta_h1 = mu2**2
                return (theta_h0, theta_h1)
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(y), grad(q))*dx
                return (a0,)
            elif term == "a*":
                as0 = inner(grad(z), grad(p))*dx
                return (as0,)
            elif term == "c":
                c0 = u*q*dx
                return (c0,)
            elif term == "c*":
                cs0 = v*p*dx
                return (cs0,)
            elif term == "m":
                m0 = y*z*dx
                return (m0,)
            elif term == "n":
                n0 = u*v*dx
                return (n0,)
            elif term == "f":
                f0 = ff*q*dx
                return (f0,)
            elif term == "g":
                g0 = y_d*z*dx(1)
                g1 = y_d*z*dx(2)
                return (g0, g1)
            elif term == "h":
                h0 = y_d*y_d*dx(1, domain=mesh)
                h1 = y_d*y_d*dx(2, domain=mesh)
                return (h0, h1)
            else:
                raise ValueError("Invalid term for assemble_operator().")
                
    # Check forms
    problem_on_reference_domain = EllipticOptimalControlOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = EllipticOptimalControlPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        as_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a*")
        as_pull_back = theta_times_operator(problem_pull_back, "a*")
        assert forms_are_close(as_on_reference_domain, as_pull_back)
        
        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)
        
        cs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c*")
        cs_pull_back = theta_times_operator(problem_pull_back, "c*")
        assert forms_are_close(cs_on_reference_domain, cs_pull_back)
        
        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)
        
        n_on_reference_domain = theta_times_operator(problem_on_reference_domain, "n")
        n_pull_back = theta_times_operator(problem_pull_back, "n")
        assert forms_are_close(n_on_reference_domain, n_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)
        
        h_on_reference_domain = theta_times_operator(problem_on_reference_domain, "h")
        h_pull_back = theta_times_operator(problem_pull_back, "h")
        assert forms_are_close(h_on_reference_domain, h_pull_back)
        
# Test forms pull back to reference domain for tutorial 19
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes_optimal_control_1(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "stokes_optimal_control_1.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "stokes_optimal_control_1_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "stokes_optimal_control_1_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("x[0]", "mu[0]*x[1]") # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    velocity_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    pressure_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(velocity_element, pressure_element, velocity_element, velocity_element, pressure_element)
    V = FunctionSpace(mesh, element, components=[["v", "s"], "p", "u", ["w", "r"], "q"])
    trial = TrialFunction(V)
    (v, p, u, w, q) = split(trial)
    test = TestFunction(V)
    (psi, pi, tau, phi, xi) = split(test)
    dx = Measure("dx")(subdomain_data=subdomains)
    alpha = 0.008
    nu = 0.1
    vx_d = Expression("x[1]", degree=1)
    ll = Constant(1.0)
    
    # Define base problem
    class StokesOptimalControl(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1.0, 1.0)
            self.mu_range = [(0.5, 2.0), (0.5, 1.5)]
            
        def init(self):
            pass
    
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOptimalControlOnReferenceDomain(StokesOptimalControl):
        def __init__(self, V, **kwargs):
            StokesOptimalControl.__init__(self, "StokesOptimalControlOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu1 = self.mu[0]
            mu2 = self.mu[1]
            if term in ("a", "a*"):
                theta_a0 = nu*mu1
                theta_a1 = nu/mu1
                return (theta_a0, theta_a1)
            elif term in ("b", "b*", "bt", "bt*"):
                theta_b0 = mu1
                theta_b1 = 1.0
                return (theta_b0, theta_b1)
            elif term in ("c", "c*"):
                theta_c0 = mu1
                return (theta_c0,)
            elif term == "m":
                theta_m0 = mu1
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha*mu1
                return (theta_n0,)
            elif term == "f":
                theta_f0 = - mu1*mu2
                return (theta_f0,)
            elif term == "g":
                theta_g0 = mu1**2
                return (theta_g0,)
            elif term == "l":
                theta_l0 = mu1
                return (theta_l0,)
            elif term == "h":
                theta_h0 = mu1**3/3.
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")

        def assemble_operator(self, term):
            if term == "a":
                a0 = v[0].dx(0)*phi[0].dx(0)*dx + v[1].dx(0)*phi[1].dx(0)*dx
                a1 = v[0].dx(1)*phi[0].dx(1)*dx + v[1].dx(1)*phi[1].dx(1)*dx
                return (a0, a1)
            elif term == "a*":
                as0 = psi[0].dx(0)*w[0].dx(0)*dx + psi[1].dx(0)*w[1].dx(0)*dx
                as1 = psi[0].dx(1)*w[0].dx(1)*dx + psi[1].dx(1)*w[1].dx(1)*dx
                return (as0, as1)
            elif term == "b":
                b0 = - xi*v[0].dx(0)*dx
                b1 = - xi*v[1].dx(1)*dx
                return (b0, b1)
            elif term == "bt":
                bt0 = - p*phi[0].dx(0)*dx
                bt1 = - p*phi[1].dx(1)*dx
                return (bt0, bt1)
            elif term == "b*":
                bs0 = - pi*w[0].dx(0)*dx
                bs1 = - pi*w[1].dx(1)*dx
                return (bs0, bs1)
            elif term == "bt*":
                bts0 = - q*psi[0].dx(0)*dx
                bts1 = - q*psi[1].dx(1)*dx
                return (bts0, bts1)
            elif term == "c":
                c0 = inner(u, phi)*dx
                return (c0,)
            elif term == "c*":
                cs0 = inner(tau, w)*dx
                return (cs0,)
            elif term == "m":
                m0 = v[0]*psi[0]*dx
                return (m0,)
            elif term == "n":
                n0 = inner(u, tau)*dx
                return (n0,)
            elif term == "f":
                f0 = phi[1]*dx
                return (f0,)
            elif term == "g":
                g0 = vx_d*psi[0]*dx
                return (g0,)
            elif term == "l":
                l0 = ll*xi*dx
                return (l0,)
            elif term == "h":
                h0 = 1.0
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")

    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "a*", "b", "b*", "bt", "bt*", "c", "c*", "m", "n", "f", "g", "h", "l", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesOptimalControlPullBack(StokesOptimalControl):
        def __init__(self, V, **kwargs):
            StokesOptimalControl.__init__(self, "StokesOptimalControlPullBack")
            self.V = V
            
        def compute_theta(self, term):
            mu2 = self.mu[1]
            if term in ("a", "a*"):
                theta_a0 = nu*1.0
                return (theta_a0,)
            elif term in ("b", "b*", "bt", "bt*"):
                theta_b0 = 1.0
                return (theta_b0,)
            elif term in ("c", "c*"):
                theta_c0 = 1.0
                return (theta_c0,)
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0,)
            elif term == "n":
                theta_n0 = alpha*1.0
                return (theta_n0,)
            elif term == "f":
                theta_f0 = - mu2
                return (theta_f0,)
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0,)
            elif term == "l":
                theta_l0 = 1.0
                return (theta_l0,)
            elif term == "h":
                theta_h0 = 1.0
                return (theta_h0,)
            else:
                raise ValueError("Invalid term for compute_theta().")
        
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(v), grad(phi))*dx
                return (a0,)
            elif term == "a*":
                ad0 = inner(grad(w), grad(psi))*dx
                return (ad0,)
            elif term == "b":
                b0 = -xi*div(v)*dx
                return (b0,)
            elif term == "b*":
                btd0 = -pi*div(w)*dx
                return (btd0,)
            elif term == "bt":
                bt0 = -p*div(phi)*dx
                return (bt0,)
            elif term == "bt*":
                bd0 = -q*div(psi)*dx
                return (bd0,)
            elif term == "c":
                c0 = inner(u, phi)*dx
                return (c0,)
            elif term == "c*":
                cd0 = inner(tau, w)*dx
                return (cd0,)
            elif term == "m":
                m0 = v[0]*psi[0]*dx
                return (m0,)
            elif term == "n":
                n0 = inner(u, tau)*dx
                return (n0,)
            elif term == "f":
                f0 = phi[1]*dx
                return (f0,)
            elif term == "g":
                g0 = vx_d*psi[0]*dx
                return (g0,)
            elif term == "l":
                l0 = ll*xi*dx
                return (l0,)
            elif term == "h":
                h0 = vx_d*vx_d*dx(domain=mesh)
                return (h0,)
            else:
                raise ValueError("Invalid term for assemble_operator().")
    # Check forms
    problem_on_reference_domain = StokesOptimalControlOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesOptimalControlPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        as_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a*")
        as_pull_back = theta_times_operator(problem_pull_back, "a*")
        assert forms_are_close(as_on_reference_domain, as_pull_back)
        
        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)
        
        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)
        
        bs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b*")
        bs_pull_back = theta_times_operator(problem_pull_back, "b*")
        assert forms_are_close(bs_on_reference_domain, bs_pull_back)
        
        bts_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt*")
        bts_pull_back = theta_times_operator(problem_pull_back, "bt*")
        assert forms_are_close(bts_on_reference_domain, bts_pull_back)
        
        c_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c")
        c_pull_back = theta_times_operator(problem_pull_back, "c")
        assert forms_are_close(c_on_reference_domain, c_pull_back)
        
        cs_on_reference_domain = theta_times_operator(problem_on_reference_domain, "c*")
        cs_pull_back = theta_times_operator(problem_pull_back, "c*")
        assert forms_are_close(cs_on_reference_domain, cs_pull_back)
        
        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)
        
        n_on_reference_domain = theta_times_operator(problem_on_reference_domain, "n")
        n_pull_back = theta_times_operator(problem_pull_back, "n")
        assert forms_are_close(n_on_reference_domain, n_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)
        
        l_on_reference_domain = theta_times_operator(problem_on_reference_domain, "l")
        l_pull_back = theta_times_operator(problem_pull_back, "l")
        assert forms_are_close(l_on_reference_domain, l_pull_back)
        
        h_on_reference_domain = theta_times_operator(problem_on_reference_domain, "h")
        h_pull_back = theta_times_operator(problem_pull_back, "h")
        assert forms_are_close(h_on_reference_domain, h_pull_back)
        
# Test forms pull back to reference domain for tutorial 23
@check_affine_and_non_affine_shape_parametrizations()
def test_pull_back_to_reference_domain_stokes_unsteady(shape_parametrization_preprocessing, AdditionalProblemDecorator, ExceptionType, exception_message):
    # Read the mesh for this problem
    mesh = Mesh(os.path.join(data_dir, "cavity.xml"))
    subdomains = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_physical_region.xml"))
    boundaries = MeshFunction("size_t", mesh, os.path.join(data_dir, "cavity_facet_region.xml"))
    
    # Define shape parametrization
    shape_parametrization_expression = [
        ("mu[0]*x[0]", "x[1]"), # subdomain 1
    ]
    shape_parametrization_expression = shape_parametrization_preprocessing(shape_parametrization_expression)
    
    # Define function space, test/trial functions, measures, auxiliary expressions
    element_u = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_u, element_p)
    V = FunctionSpace(mesh, element, components=[["u", "s"], "p"])
    up = TrialFunction(V)
    (u, p) = split(up)
    vq = TestFunction(V)
    (v, q) = split(vq)
    dx = Measure("dx")(subdomain_data=subdomains)
    
    ff = Constant((1.0, 2.0))
    gg = Constant(1.0)
    
    # Define base problem
    class StokesUnsteady(ParametrizedProblem):
        def __init__(self, folder_prefix):
            ParametrizedProblem.__init__(self, folder_prefix)
            self.mu = (1., )
            self.mu_range = [(0.5, 2.5)]
            
        def init(self):
            pass
            
    # Define problem with forms written on reference domain
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesUnsteadyOnReferenceDomain(StokesUnsteady):
        def __init__(self, V, **kwargs):
            StokesUnsteady.__init__(self, "StokesUnsteadyOnReferenceDomain")
            self.V = V
            
        def compute_theta(self, term):
            mu = self.mu
            mu1 = mu[0]
            if term == "a":
                theta_a0 = 1./mu1
                theta_a1 = mu1
                return (theta_a0, theta_a1)
            elif term in ("b", "bt"):
                theta_b0 = 1.
                theta_b1 = mu1
                return (theta_b0, theta_b1)
            elif term == "f":
                theta_f0 = mu1
                return (theta_f0, )
            elif term == "g":
                theta_g0 = mu1
                return (theta_g0, )
            elif term == "m":
                theta_m0 = mu1
                return (theta_m0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
            
        def assemble_operator(self, term):
            if term == "a":
                a0 = (u[0].dx(0)*v[0].dx(0) + u[1].dx(0)*v[1].dx(0))*dx
                a1 = (u[0].dx(1)*v[0].dx(1) + u[1].dx(1)*v[1].dx(1))*dx
                return (a0, a1)
            elif term == "b":
                b0 = - q*u[0].dx(0)*dx
                b1 = - q*u[1].dx(1)*dx
                return (b0, b1)
            elif term == "bt":
                bt0 = - p*v[0].dx(0)*dx
                bt1 = - p*v[1].dx(1)*dx
                return (bt0, bt1)
            elif term == "f":
                f0 = inner(ff, v)*dx
                return (f0, )
            elif term == "g":
                g0 = gg*q*dx
                return (g0, )
            elif term == "m":
                m0 = inner(u, v)*dx
                return (m0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Define problem with forms pulled back reference domain
    @AdditionalProblemDecorator()
    @PullBackFormsToReferenceDomain("a", "b", "bt", "m", "f", "g", debug=True)
    @ShapeParametrization(*shape_parametrization_expression)
    class StokesUnsteadyPullBack(StokesUnsteady):
        def __init__(self, V, **kwargs):
            StokesUnsteady.__init__(self, "StokesUnsteadyPullBack")
            self.V = V
            
        def compute_theta(self, term):
            if term == "a":
                theta_a0 = 1.0
                return (theta_a0, )
            elif term in ("b", "bt"):
                theta_b0 = 1.0
                return (theta_b0, )
            elif term == "f":
                theta_f0 = 1.0
                return (theta_f0, )
            elif term == "g":
                theta_g0 = 1.0
                return (theta_g0, )
            elif term == "m":
                theta_m0 = 1.0
                return (theta_m0, )
            else:
                raise ValueError("Invalid term for compute_theta().")
                
        def assemble_operator(self, term):
            if term == "a":
                a0 = inner(grad(u), grad(v))*dx
                return (a0, )
            elif term == "b":
                b0 = - q*div(u)*dx
                return (b0, )
            elif term == "bt":
                bt0 = - p*div(v)*dx
                return (bt0, )
            elif term == "f":
                f0 = inner(ff, v)*dx
                return (f0, )
            elif term == "g":
                g0 = q*gg*dx
                return (g0, )
            elif term == "m":
                m0 = inner(u, v)*dx
                return (m0, )
            else:
                raise ValueError("Invalid term for assemble_operator().")
    
    # Check forms
    problem_on_reference_domain = StokesUnsteadyOnReferenceDomain(V, subdomains=subdomains, boundaries=boundaries)
    problem_pull_back = StokesUnsteadyPullBack(V, subdomains=subdomains, boundaries=boundaries)
    problem_on_reference_domain.init()
    problem_pull_back.init()
    for mu in itertools.product(*problem_on_reference_domain.mu_range):
        problem_on_reference_domain.set_mu(mu)
        problem_pull_back.set_mu(mu)
        
        a_on_reference_domain = theta_times_operator(problem_on_reference_domain, "a")
        a_pull_back = theta_times_operator(problem_pull_back, "a")
        assert forms_are_close(a_on_reference_domain, a_pull_back)
        
        b_on_reference_domain = theta_times_operator(problem_on_reference_domain, "b")
        b_pull_back = theta_times_operator(problem_pull_back, "b")
        assert forms_are_close(b_on_reference_domain, b_pull_back)
        
        bt_on_reference_domain = theta_times_operator(problem_on_reference_domain, "bt")
        bt_pull_back = theta_times_operator(problem_pull_back, "bt")
        assert forms_are_close(bt_on_reference_domain, bt_pull_back)
        
        f_on_reference_domain = theta_times_operator(problem_on_reference_domain, "f")
        f_pull_back = theta_times_operator(problem_pull_back, "f")
        assert forms_are_close(f_on_reference_domain, f_pull_back)
        
        g_on_reference_domain = theta_times_operator(problem_on_reference_domain, "g")
        g_pull_back = theta_times_operator(problem_pull_back, "g")
        assert forms_are_close(g_on_reference_domain, g_pull_back)
        
        m_on_reference_domain = theta_times_operator(problem_on_reference_domain, "m")
        m_pull_back = theta_times_operator(problem_pull_back, "m")
        assert forms_are_close(m_on_reference_domain, m_pull_back)
