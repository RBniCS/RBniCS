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
