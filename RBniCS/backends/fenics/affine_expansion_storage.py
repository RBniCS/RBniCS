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
## @file affine_expansion_online_storage.py
#  @brief Type for storing online quantities related to an affine expansion
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from dolfin import assemble, DirichletBC
from RBniCS.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override, tuple_of

@Extends(AbstractAffineExpansionStorage)
@BackendFor("fenics", inputs=((tuple_of(list_of(DirichletBC)), tuple_of(Form), tuple_of(Function.Type()), tuple_of(Matrix.Type()), tuple_of(Vector.Type()), tuple_of((Form, Matrix.Type())), tuple_of((Form, Vector.Type()))), ))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args):
        self._content = None
        self._type = None
        # Type checking
        is_Form = self._is_Form(args[0])
        is_Tensor = self._is_Tensor(args[0])
        is_DirichletBC = self._is_DirichletBC(args[0])
        is_Function = self._is_Function(args[0])
        assert is_Form or is_DirichletBC or is_Function or is_Tensor
        all_is_Form = list()
        all_is_Tensor = list()
        all_is_parametrized = list()
        for i in range(len(args)):
            if is_Form or is_Tensor:
                all_is_Form.append(self._is_Form(args[i]))
                all_is_Tensor.append(self._is_Tensor(args[i]))
                assert all_is_Form[i] or all_is_Tensor[i]
                if all_is_Form[i]:
                    all_is_parametrized.append(ParametrizedTensorFactory(None, args[i]).is_parametrized())
                else:
                    all_is_parametrized.append(False)
            elif is_DirichletBC:
                assert self._is_DirichletBC(args[i])
            elif is_Function:
                assert self._is_Function(args[i])
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
        # Actual init
        if is_Form or is_Tensor:
            is_parametrized = any(all_is_parametrized)
            if not is_parametrized:
                # keep_diagonal is enabled because it is needed to constrain DirichletBC eigenvalues in SCM
                self._content = list()
                for (arg, arg_is_Form) in zip(args, all_is_Form):
                    if arg_is_Form:
                        self._content.append(assemble(arg, keep_diagonal=True))
                    else:
                        self._content.append(arg)
                self._type = "AssembledForm"
            else:
                assert all(all_is_Form)
                self._content = args
                self._type = "Form"
        elif is_DirichletBC:
            self._content = args
            self._type = "DirichletBC"
        elif is_Function:
            self._content = args
            self._type = "Function"
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
    def _is_Function(arg):
        return isinstance(arg, Function.Type())

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
        
