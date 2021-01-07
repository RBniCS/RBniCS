# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from sympy import ccode, MatrixSymbol, sympify
from mpi4py.MPI import MAX, MIN
from dolfin import ALE, cells, Function, FunctionSpace, LagrangeInterpolator, VectorFunctionSpace
from dolfin.cpp.mesh import MeshFunctionSizet
from rbnics.backends.abstract import MeshMotion as AbstractMeshMotion
from rbnics.backends.dolfin.wrapping import ParametrizedExpression
from rbnics.utils.decorators import BackendFor, tuple_of


@BackendFor("dolfin", inputs=(FunctionSpace, MeshFunctionSizet, tuple_of(tuple_of(str))))
class MeshMotion(AbstractMeshMotion):
    def __init__(self, V, subdomains, shape_parametrization_expression):
        # Store dolfin data structure related to the geometrical parametrization
        self.mesh = subdomains.mesh()
        self.subdomains = subdomains
        self.reference_coordinates = self.mesh.coordinates().copy()
        self.deformation_V = VectorFunctionSpace(self.mesh, "Lagrange", 1)
        self.subdomain_id_to_deformation_dofs = dict()  # from int to list
        for cell in cells(self.mesh):
            subdomain_id = int(self.subdomains[cell]) - 1  # tuple start from 0, while subdomains from 1
            if subdomain_id not in self.subdomain_id_to_deformation_dofs:
                self.subdomain_id_to_deformation_dofs[subdomain_id] = list()
            dofs = self.deformation_V.dofmap().cell_dofs(cell.index())
            for dof in dofs:
                global_dof = self.deformation_V.dofmap().local_to_global_index(dof)
                if (self.deformation_V.dofmap().ownership_range()[0] <= global_dof
                        and global_dof < self.deformation_V.dofmap().ownership_range()[1]):
                    self.subdomain_id_to_deformation_dofs[subdomain_id].append(dof)
        # In parallel some subdomains may not be present on all processors. Fill in
        # the dict with empty lists if that is the case
        mpi_comm = self.mesh.mpi_comm()
        min_subdomain_id = mpi_comm.allreduce(min(self.subdomain_id_to_deformation_dofs.keys()), op=MIN)
        max_subdomain_id = mpi_comm.allreduce(max(self.subdomain_id_to_deformation_dofs.keys()), op=MAX)
        for subdomain_id in range(min_subdomain_id, max_subdomain_id + 1):
            if subdomain_id not in self.subdomain_id_to_deformation_dofs:
                self.subdomain_id_to_deformation_dofs[subdomain_id] = list()
        # Subdomain numbering is contiguous
        assert min(self.subdomain_id_to_deformation_dofs.keys()) == 0
        assert len(self.subdomain_id_to_deformation_dofs.keys()) == (
            max(self.subdomain_id_to_deformation_dofs.keys()) + 1)

        # Store the shape parametrization expression
        self.shape_parametrization_expression = shape_parametrization_expression
        assert len(self.shape_parametrization_expression) == len(self.subdomain_id_to_deformation_dofs.keys())

        # Prepare storage for displacement expression, computed by init()
        self.displacement_expression = list()

    def init(self, problem):
        if len(self.displacement_expression) == 0:  # avoid initialize multiple times
            # Preprocess the shape parametrization expression to convert it in the displacement expression
            # This cannot be done during __init__ because at construction time the number
            # of parameters is still unknown

            # Declare first some sympy simbolic quantities, needed by ccode
            from rbnics.shape_parametrization.utils.symbolic import sympy_symbolic_coordinates
            x = sympy_symbolic_coordinates(self.mesh.geometry().dim(), MatrixSymbol)
            mu = MatrixSymbol("mu", len(problem.mu), 1)

            # Then carry out the proprocessing
            for shape_parametrization_expression_on_subdomain in self.shape_parametrization_expression:
                displacement_expression_on_subdomain = list()
                assert len(shape_parametrization_expression_on_subdomain) == self.mesh.geometry().dim()
                for (component, shape_parametrization_component_on_subdomain) in enumerate(
                        shape_parametrization_expression_on_subdomain):
                    # convert from shape parametrization T to displacement d = T - I
                    displacement_expression_component_on_subdomain = sympify(
                        shape_parametrization_component_on_subdomain + " - x[" + str(component) + "]",
                        locals={"x": x, "mu": mu})
                    displacement_expression_on_subdomain.append(
                        ccode(displacement_expression_component_on_subdomain).replace(", 0]", "]"),
                    )
                self.displacement_expression.append(
                    ParametrizedExpression(
                        problem,
                        tuple(displacement_expression_on_subdomain),
                        mu=problem.mu,
                        element=self.deformation_V.ufl_element(),
                        domain=self.mesh
                    )
                )

    def move_mesh(self):
        displacement = self.compute_displacement()
        ALE.move(self.mesh, displacement)

    def reset_reference(self):
        self.mesh.coordinates()[:] = self.reference_coordinates

    # Auxiliary method to deform the domain
    def compute_displacement(self):
        displacement = Function(self.deformation_V)
        assert len(self.displacement_expression) == len(self.shape_parametrization_expression)
        for (subdomain, displacement_expression_on_subdomain) in enumerate(self.displacement_expression):
            displacement_function_on_subdomain = Function(self.deformation_V)
            LagrangeInterpolator.interpolate(displacement_function_on_subdomain, displacement_expression_on_subdomain)
            subdomain_dofs = self.subdomain_id_to_deformation_dofs[subdomain]
            displacement.vector()[subdomain_dofs] = displacement_function_on_subdomain.vector()[subdomain_dofs]
        return displacement
