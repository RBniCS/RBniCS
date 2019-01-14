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

from slepc4py import SLEPc

def BasicSLEPcEPSSolver(backend, wrapping):
    class _BasicSLEPcEPSSolver(object):
        def __init__(self, A, B):
            self.eps = SLEPc.EPS().create(wrapping.get_mpi_comm(A))
            if B is not None:
                self.A = wrapping.to_petsc4py(A)
                self.B = wrapping.to_petsc4py(B)
                self.eps.setOperators(self.A, self.B)
            else:
                self.A = wrapping.to_petsc4py(A)
                self.B = None
                self.eps.setOperators(self.A)
            
        def set_parameters(self, parameters):
            eps_tolerances = [None, None]
            for (key, value) in parameters.items():
                if key == "linear_solver":
                    ksp = self.eps.getST().getKSP()
                    ksp.setType("preonly")
                    ksp.getPC().setType("lu")
                    if value == "default":
                        value = wrapping.get_default_linear_solver()
                    if hasattr(ksp.getPC(), "setFactorSolverType"): # PETSc >= 3.9
                        ksp.getPC().setFactorSolverType(value)
                    else:
                        ksp.getPC().setFactorSolverPackage(value)
                elif key == "maximum_iterations":
                    eps_tolerances[1] = value
                elif key == "problem_type":
                    if value == "hermitian":
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
                    elif value == "non_hermitian":
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
                    elif value == "gen_hermitian":
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
                    elif value == "gen_non_hermitian":
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
                    elif value == "pos_gen_non_hermitian":
                        self.eps.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
                    else:
                        raise RuntimeError("Invalid problem type")
                elif key == "solver":
                    if value == "power":
                        self.eps.setType(SLEPc.EPS.Type.POWER)
                    elif value == "subspace":
                        self.eps.setType(SLEPc.EPS.Type.SUBSPACE)
                    elif value == "arnoldi":
                        self.eps.setType(SLEPc.EPS.Type.ARNOLDI)
                    elif value == "lanczos":
                        self.eps.setType(SLEPc.EPS.Type.LANCZOS)
                    elif value == "krylov-schur":
                        self.eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
                    elif value == "lapack":
                        self.eps.setType(SLEPc.EPS.Type.LAPACK)
                    elif value == "arpack":
                        self.eps.setType(SLEPc.EPS.Type.ARPACK)
                    elif value == "jacobi-davidson":
                        self.eps.setType(SLEPc.EPS.Type.JD)
                    elif value == "generalized-davidson":
                        self.eps.setType(SLEPc.EPS.Type.GD)
                    else:
                        raise RuntimeError("Invalid solver type")
                elif key == "spectral_shift":
                    st = self.eps.getST()
                    st.setShift(value)
                elif key == "spectral_transform":
                    assert value == "shift-and-invert"
                    st = self.eps.getST()
                    st.setType(SLEPc.ST.Type.SINVERT)
                elif key == "spectrum":
                    if value == "largest magnitude":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
                    elif value == "smallest magnitude":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
                    elif value == "largest real":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
                    elif value == "smallest real":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
                    elif value == "largest imaginary":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
                    elif value == "smallest imaginary":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_IMAGINARY)
                    elif value == "target magnitude":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
                        if "spectral_shift" in parameters:
                            self.eps.setTarget(parameters["spectral_shift"])
                    elif value == "target real":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
                        if "spectral_shift" in parameters:
                            self.eps.setTarget(parameters["spectral_shift"])
                    elif value == "target imaginary":
                        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY)
                        if "spectral_shift" in parameters:
                            self.eps.setTarget(parameters["spectral_shift"])
                    else:
                        raise RuntimeError("Invalid spectrum type")
                elif key == "tolerance":
                    eps_tolerances[0] = value
                else:
                    raise RuntimeError("Invalid paramater passed to SLEPc EPS object.")
            self.eps.setTolerances(*eps_tolerances)
            # Finally, read in additional options from the command line
            self.eps.setFromOptions()

        def solve(self, n_eigs=None):
            if n_eigs is None:
                n_eigs = self.A.getSize()[0]
            self.eps.setDimensions(n_eigs)
            self.eps.solve()
            assert self.eps.getConverged() >= n_eigs
                
        def get_eigenvalue(self, i):
            assert i < self.eps.getConverged()
            eig_i = self.eps.getEigenvalue(i)
            return eig_i.real, eig_i.imag
        
        def get_eigenvector(self, i, eigv_i_real, eigv_i_imag):
            assert i < self.eps.getConverged()
            eigv_i_real_petsc4py = wrapping.to_petsc4py(eigv_i_real)
            eigv_i_imag_petsc4py = wrapping.to_petsc4py(eigv_i_imag)
            self.eps.getEigenvector(i, eigv_i_real_petsc4py, eigv_i_imag_petsc4py)
            return (eigv_i_real, eigv_i_imag)
    
    return _BasicSLEPcEPSSolver
