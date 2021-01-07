# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import compile_cpp_code

cpp_code = """
    #include <petscksp.h>
    #include <dolfin/common/MPI.h>
    #include <pybind11/pybind11.h>

    std::string get_default_linear_solver()
    {
        if (dolfin::MPI::size(MPI_COMM_WORLD) == 1)
        {
            #if PETSC_HAVE_UMFPACK || PETSC_HAVE_SUITESPARSE
            return "umfpack";
            #elif PETSC_HAVE_MUMPS
            return "mumps";
            #elif PETSC_HAVE_PASTIX
            return "pastix";
            #elif PETSC_HAVE_SUPERLU
            return "superlu";
            #elif PETSC_HAVE_SUPERLU_DIST
            return "superlu_dist";
            #else
            throw std::logic_error("No suitable solver for serial LU found");
            #endif
        }
        else
        {
            #if PETSC_HAVE_MUMPS
            return "mumps";
            #elif PETSC_HAVE_SUPERLU_DIST
            return "superlu_dist";
            #elif PETSC_HAVE_PASTIX
            return "pastix";
            #else
            throw std::logic_error("No suitable solver for parallel LU found");
            #endif
        }
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
        m.def("get_default_linear_solver", &get_default_linear_solver);
    }
    """
get_default_linear_solver = compile_cpp_code(cpp_code).get_default_linear_solver
