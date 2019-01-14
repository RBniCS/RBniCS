# Copyright (C) 2015-2019 by the RBniCS authors
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

from dolfin import as_backend_type, compile_cpp_code
    
def matrix_mul_vector(matrix, vector):
    return matrix*vector
    
cpp_code = """
    #include <pybind11/pybind11.h>
    #include <dolfin/la/PETScMatrix.h>
    
    void throw_error(PetscErrorCode ierr, std::string reason);
    
    PetscScalar vectorized_matrix_inner_vectorized_matrix(std::shared_ptr<dolfin::PETScMatrix> A, std::shared_ptr<dolfin::PETScMatrix> B)
    {
        Mat a = A->mat();
        Mat b = B->mat();
        PetscInt start_a, end_a, ncols_a, start_b, end_b, ncols_b;
        PetscErrorCode ierr;
        const PetscInt *cols_a, *cols_b;
        const PetscScalar *vals_a, *vals_b;
        PetscScalar output(0.);

        ierr = MatGetOwnershipRange(a, &start_a, &end_a);
        if (ierr != 0) throw_error(ierr, "MatGetOwnershipRange");
        ierr = MatGetOwnershipRange(b, &start_b, &end_b);
        if (ierr != 0) throw_error(ierr, "MatGetOwnershipRange");
        if (start_a != start_b) throw_error(ierr, "start_a != start_b");
        if (end_a != end_b) throw_error(ierr, "end_a != end_b");
        for (PetscInt i(start_a); i < end_a; i++) {
            ierr = MatGetRow(a, i, &ncols_a, &cols_a, &vals_a);
            if (ierr != 0) throw_error(ierr, "MatGetRow");
            ierr = MatGetRow(b, i, &ncols_b, &cols_b, &vals_b);
            if (ierr != 0) throw_error(ierr, "MatGetRow");
            if (ncols_a != ncols_b) throw_error(ierr, "ncols_a != ncols_b");
            for (PetscInt j(0); j < ncols_a; j++) {
                if (cols_a[j] != cols_b[j]) throw_error(ierr, "cols_a[j] != cols_b[j]");
                output += vals_a[j]*vals_b[j];
            }
            ierr = MatRestoreRow(a, i, &ncols_a, &cols_a, &vals_a);
            if (ierr != 0) throw_error(ierr, "MatRestoreRow");
            ierr = MatRestoreRow(b, i, &ncols_b, &cols_b, &vals_b);
            if (ierr != 0) throw_error(ierr, "MatRestoreRow");
        }
        return output;
    }
    
    void throw_error(PetscErrorCode ierr, std::string reason)
    {
        throw std::runtime_error("Error in vectorized_matrix_inner_vectorized_matrix: reason " + reason + ", error code " + std::to_string(ierr));
    }
    
    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("vectorized_matrix_inner_vectorized_matrix", &vectorized_matrix_inner_vectorized_matrix);
    }
"""
_vectorized_matrix_inner_vectorized_matrix = compile_cpp_code(cpp_code).vectorized_matrix_inner_vectorized_matrix

def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    return _vectorized_matrix_inner_vectorized_matrix(as_backend_type(matrix), as_backend_type(other_matrix))
