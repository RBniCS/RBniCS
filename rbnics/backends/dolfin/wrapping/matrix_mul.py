# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import compile_cpp_code


def matrix_mul_vector(matrix, vector):
    return matrix * vector


cpp_code = """
    #include <pybind11/pybind11.h>
    #include <dolfin/la/LinearAlgebraObject.h>
    #include <dolfin/la/GenericMatrix.h>
    #include <dolfin/la/PETScMatrix.h>

    void throw_error(PetscErrorCode ierr, std::string reason);

    PetscScalar vectorized_matrix_inner_vectorized_matrix(std::shared_ptr<dolfin::GenericMatrix> A,
                                                          std::shared_ptr<dolfin::GenericMatrix> B)
    {
        Mat a = dolfin::as_type<dolfin::PETScMatrix>(*A).mat();
        Mat b = dolfin::as_type<dolfin::PETScMatrix>(*B).mat();
        PetscInt start_a, end_a, ncols_a, start_b, end_b, ncols_b;
        PetscErrorCode ierr;
        const PetscInt *cols_a, *cols_b;
        const PetscScalar *vals_a, *vals_b;
        PetscScalar sum(0.);

        ierr = MatGetOwnershipRange(a, &start_a, &end_a);
        if (ierr != 0) throw_error(ierr, "MatGetOwnershipRange");
        if (a != b) {
            ierr = MatGetOwnershipRange(b, &start_b, &end_b);
            if (ierr != 0) throw_error(ierr, "MatGetOwnershipRange");
        }
        else {
            start_b = start_a;
            end_b = end_a;
        }
        if (start_a != start_b) throw_error(ierr, "start_a != start_b");
        if (end_a != end_b) throw_error(ierr, "end_a != end_b");

        for (PetscInt i(start_a); i < end_a; i++) {
            ierr = MatGetRow(a, i, &ncols_a, &cols_a, &vals_a);
            if (ierr != 0) throw_error(ierr, "MatGetRow");
            if (a != b) {
                ierr = MatGetRow(b, i, &ncols_b, &cols_b, &vals_b);
                if (ierr != 0) throw_error(ierr, "MatGetRow");
                if (ncols_a != ncols_b) throw_error(ierr, "ncols_a != ncols_b");
            }
            else {
                ncols_b = ncols_a;
                cols_b = cols_a;
                vals_b = vals_a;
            }
            for (PetscInt j(0); j < ncols_a; j++) {
                if (cols_a[j] != cols_b[j]) throw_error(ierr, "cols_a[j] != cols_b[j]");
                sum += vals_a[j]*vals_b[j];
            }
            if (a != b) {
                ierr = MatRestoreRow(b, i, &ncols_b, &cols_b, &vals_b);
                if (ierr != 0) throw_error(ierr, "MatRestoreRow");
            }
            else {
                ncols_b = 0;
                cols_b = NULL;
                vals_b = NULL;
            }
            ierr = MatRestoreRow(a, i, &ncols_a, &cols_a, &vals_a);
            if (ierr != 0) throw_error(ierr, "MatRestoreRow");
        }

        PetscReal output(0.);
        ierr = MPIU_Allreduce(&sum, &output, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) a));
        if (ierr != 0) throw_error(ierr, "MPIU_Allreduce");
        return output;
    }

    void throw_error(PetscErrorCode ierr, std::string reason)
    {
        throw std::runtime_error("Error in vectorized_matrix_inner_vectorized_matrix: reason " + reason
                                 + ", error code " + std::to_string(ierr));
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("vectorized_matrix_inner_vectorized_matrix", &vectorized_matrix_inner_vectorized_matrix);
    }
"""
_vectorized_matrix_inner_vectorized_matrix = compile_cpp_code(cpp_code).vectorized_matrix_inner_vectorized_matrix


def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    return _vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix)
