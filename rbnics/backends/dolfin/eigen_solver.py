# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from petsc4py import PETSc
from ufl import Form
from dolfin import (__version__ as dolfin_version, as_backend_type, assemble, compile_cpp_code, DirichletBC,
                    Function, FunctionSpace, PETScMatrix, PETScVector, SLEPcEigenSolver)
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.wrapping.dirichlet_bc import ProductOutputDirichletBC
from rbnics.backends.abstract import EigenSolver as AbstractEigenSolver
from rbnics.utils.decorators import BackendFor, dict_of, list_of, overload


@BackendFor("dolfin", inputs=(FunctionSpace, (Form, Matrix.Type(), ParametrizedTensorFactory),
                              (Form, Matrix.Type(), ParametrizedTensorFactory, None),
                              (list_of(DirichletBC), ProductOutputDirichletBC, dict_of(str, list_of(DirichletBC)),
                               dict_of(str, ProductOutputDirichletBC), None)))
class EigenSolver(AbstractEigenSolver):
    def __init__(self, V, A, B=None, bcs=None):
        self.V = V
        if bcs is not None:
            self._set_boundary_conditions(bcs)
        A = self._assemble_if_form(A)
        if B is not None:
            B = self._assemble_if_form(B)
        self._set_operators(A, B)
        if self.B is not None:
            self.eigen_solver = SLEPcEigenSolver(self.condensed_A, self.condensed_B)
        else:
            self.eigen_solver = SLEPcEigenSolver(self.condensed_A)

    @staticmethod
    @overload
    def _assemble_if_form(mat: Form):
        return assemble(mat, keep_diagonal=True)

    @staticmethod
    @overload
    def _assemble_if_form(mat: ParametrizedTensorFactory):
        return evaluate(mat)

    @staticmethod
    @overload
    def _assemble_if_form(mat: Matrix.Type()):
        return mat

    def _set_boundary_conditions(self, bcs):
        # List all local and constrained local dofs
        local_dofs = set()
        constrained_local_dofs = set()
        for bc in bcs:
            dofmap = bc.function_space().dofmap()
            local_range = dofmap.ownership_range()
            local_dofs.update(list(range(local_range[0], local_range[1])))
            constrained_local_dofs.update([
                dofmap.local_to_global_index(local_dof_index) for local_dof_index in bc.get_boundary_values().keys()
            ])

        # List all unconstrained dofs
        unconstrained_local_dofs = local_dofs.difference(constrained_local_dofs)
        unconstrained_local_dofs = list(unconstrained_local_dofs)

        # Generate IS accordingly
        comm = bcs[0].function_space().mesh().mpi_comm()
        for bc in bcs:
            assert comm == bc.function_space().mesh().mpi_comm()
        self._is = PETSc.IS().createGeneral(unconstrained_local_dofs, comm)

    def _set_operators(self, A, B):
        if hasattr(self, "_is"):  # there were Dirichlet BCs
            (self.A, self.condensed_A) = self._condense_matrix(A)
            if B is not None:
                (self.B, self.condensed_B) = self._condense_matrix(B)
            else:
                (self.B, self.condensed_B) = (None, None)
        else:
            (self.A, self.condensed_A) = (as_backend_type(A), as_backend_type(A))
            if B is not None:
                (self.B, self.condensed_B) = (as_backend_type(B), as_backend_type(B))
            else:
                (self.B, self.condensed_B) = (None, None)

    def _condense_matrix(self, mat):
        mat = as_backend_type(mat)

        petsc_version = PETSc.Sys().getVersionInfo()
        if petsc_version["major"] == 3 and petsc_version["minor"] <= 7:
            condensed_mat = mat.mat().getSubMatrix(self._is, self._is)
        else:
            condensed_mat = mat.mat().createSubMatrix(self._is, self._is)

        return mat, PETScMatrix(condensed_mat)

    def set_parameters(self, parameters):
        # Helper functions
        cpp_code = """
            #include <pybind11/pybind11.h>
            #include <dolfin/la/PETScLUSolver.h> // defines PCFactorSetMatSolverType macro for PETSc <= 3.8
            #include <dolfin/la/SLEPcEigenSolver.h>

            void throw_error(PetscErrorCode ierr, std::string reason);

            void set_linear_solver(std::shared_ptr<dolfin::SLEPcEigenSolver> eigen_solver, std::string lu_method)
            {
                ST st;
                KSP ksp;
                PC pc;
                PetscErrorCode ierr;

                ierr = EPSGetST(eigen_solver->eps(), &st);
                if (ierr != 0) throw_error(ierr, "EPSGetST");
                ierr = STGetKSP(st, &ksp);
                if (ierr != 0) throw_error(ierr, "STGetKSP");
                ierr = KSPGetPC(ksp, &pc);
                if (ierr != 0) throw_error(ierr, "KSPGetPC");

                ierr = STSetType(st, STSINVERT);
                if (ierr != 0) throw_error(ierr, "STSetType");
                ierr = KSPSetType(ksp, KSPPREONLY);
                if (ierr != 0) throw_error(ierr, "KSPSetType");
                ierr = PCSetType(pc, PCLU);
                if (ierr != 0) throw_error(ierr, "PCSetType");

                ierr = PCFactorSetMatSolverType(pc, lu_method.c_str());
                if (ierr != 0) throw_error(ierr, "PCFactorSetMatSolverType");
            }

            void throw_error(PetscErrorCode ierr, std::string reason)
            {
                throw std::runtime_error("Error in set_linear_solver: reason " + reason
                                         + ",error code " + std::to_string(ierr));
            }

            PYBIND11_MODULE(SIGNATURE, m)
            {
                m.def("set_linear_solver", &set_linear_solver);
            }
        """

        cpp_module = compile_cpp_code(cpp_code)
        set_linear_solver = cpp_module.set_linear_solver

        if "spectral_transform" in parameters and parameters["spectral_transform"] == "shift-and-invert":
            parameters["spectrum"] = "target real"
            if "linear_solver" in parameters:
                set_linear_solver(self.eigen_solver, parameters["linear_solver"])
                parameters.pop("linear_solver")
        self.eigen_solver.parameters.update(parameters)

    def solve(self, n_eigs=None):
        assert n_eigs is not None
        self.eigen_solver.solve(n_eigs)
        assert self.eigen_solver.get_number_converged() >= n_eigs

    def get_eigenvalue(self, i):
        assert i < self.eigen_solver.get_number_converged()
        return self.eigen_solver.get_eigenvalue(i)

    def get_eigenvector(self, i):
        assert i < self.eigen_solver.get_number_converged()

        # Initialize eigenvectors
        real_vector = PETScVector()
        imag_vector = PETScVector()
        self.A.init_vector(real_vector, 0)
        self.A.init_vector(imag_vector, 0)

        # Condense input vectors
        if hasattr(self, "_is"):  # there were Dirichlet BCs
            condensed_real_vector = PETScVector(real_vector.vec().getSubVector(self._is))
            condensed_imag_vector = PETScVector(imag_vector.vec().getSubVector(self._is))
        else:
            condensed_real_vector = real_vector
            condensed_imag_vector = imag_vector

        # Get eigenpairs
        if dolfin_version.startswith("2018.1"):  # TODO remove when 2018.2.0 is released
            # Helper functions
            cpp_code = """
                #include <pybind11/pybind11.h>
                #include <dolfin/la/PETScVector.h>
                #include <dolfin/la/SLEPcEigenSolver.h>

                void get_eigen_pair(std::shared_ptr<dolfin::SLEPcEigenSolver> eigen_solver,
                                    std::shared_ptr<dolfin::PETScVector> condensed_real_vector,
                                    std::shared_ptr<dolfin::PETScVector> condensed_imag_vector,
                                    std::size_t i)
                {
                    const PetscInt ii = static_cast<PetscInt>(i);
                    double real_value;
                    double imag_value;
                    EPSGetEigenpair(eigen_solver->eps(), ii, &real_value, &imag_value, condensed_real_vector->vec(),
                                    condensed_imag_vector->vec());
                }

                PYBIND11_MODULE(SIGNATURE, m)
                {
                    m.def("get_eigen_pair", &get_eigen_pair);
                }
            """

            get_eigen_pair = compile_cpp_code(cpp_code).get_eigen_pair
            get_eigen_pair(self.eigen_solver, condensed_real_vector, condensed_imag_vector, i)
        else:
            self.eigen_solver.get_eigenpair(condensed_real_vector, condensed_imag_vector, i)

        # Restore input vectors
        if hasattr(self, "_is"):  # there were Dirichlet BCs
            real_vector.vec().restoreSubVector(self._is, condensed_real_vector.vec())
            imag_vector.vec().restoreSubVector(self._is, condensed_imag_vector.vec())

        # Return as Function
        return (Function(self.V, real_vector), Function(self.V, imag_vector))
