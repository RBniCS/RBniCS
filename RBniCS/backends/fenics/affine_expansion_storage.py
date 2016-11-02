# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file affine_expansion_online_storage.py
#  @brief Type for storing online quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from dolfin import adjoint, assemble, DirichletBC
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override, tuple_of

@Extends(AbstractAffineExpansionStorage)
@BackendFor("FEniCS", inputs=((tuple_of(list_of(DirichletBC)), tuple_of(Form), tuple_of(Matrix.Type()), tuple_of(Vector.Type())), (bool, None)))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args, symmetrize=None):
        self._content = None
        self._type = None
        # Type checking
        is_Form = self._is_Form(args[0])
        is_DirichletBC = self._is_DirichletBC(args[0])
        is_Tensor = self._is_Tensor(args[0])
        assert is_Form or is_DirichletBC or is_Tensor
        for i in range(1, len(args)):
            if is_Form:
                assert self._is_Form(args[i])
            elif is_DirichletBC:
                assert self._is_DirichletBC(args[i])
            elif is_Tensor:
                assert self._is_Tensor(args[i])
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
        # Actual init
        if is_Form:
            if symmetrize is None or symmetrize is False:
                self._content = [assemble(arg) for arg in args]
            elif symmetrize is True:
                # this case is required by SCM. keep_diagonal is enabled because it is needed to constrain DirichletBC eigenvalues
                self._content = [assemble(0.5*(arg + adjoint(arg)), keep_diagonal=True) for arg in args]
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
            self._type = "Form"
        elif is_DirichletBC:
            assert symmetrize is None
            self._content = args
            self._type = "DirichletBC"
        elif is_Tensor:
            assert symmetrize is None
            self._content = args
            self._type = "Form"
        else:
            return TypeError("Invalid input arguments to AffineExpansionStorage")
        
    @staticmethod
    def _is_Form(arg):
        return isinstance(arg, Form)
        
    @staticmethod
    def _is_DirichletBC(arg):
        if not isinstance(arg, list):
            return False
        else:
            for bc in arg:
                if not isinstance(bc, DirichletBC):
                    return False
            return True

    @staticmethod
    def _is_Tensor(arg):
        return isinstance(arg, (Matrix.Type(), Vector.Type()))
        
    def type(self):
        return self._type
        
    @override
    def __getitem__(self, key):
        return self._content[key]
        
    @override
    def __iter__(self):
        return self._content.__iter__()
        
    @override
    def __len__(self):
        assert self._content is not None
        return len(self._content)
        
