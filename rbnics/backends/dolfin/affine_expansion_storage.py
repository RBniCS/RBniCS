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

from ufl import Form
from dolfin import assemble, DirichletBC
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.wrapping import form_iterator, is_parametrized
from rbnics.utils.config import config
from rbnics.utils.decorators import BackendFor, Extends, list_of, override, tuple_of

@Extends(AbstractAffineExpansionStorage)
@BackendFor("dolfin", inputs=((tuple_of(list_of(DirichletBC)), tuple_of(Form), tuple_of(Function.Type()), tuple_of(Matrix.Type()), tuple_of(Vector.Type()), tuple_of((Form, Matrix.Type())), tuple_of((Form, Vector.Type()))), ))
class AffineExpansionStorage(AbstractAffineExpansionStorage):
    @override
    def __init__(self, args):
        self._content = None
        self._type = None
        # Get config value
        delay_assembly = config.get("backends", "delay assembly")
        # Type checking
        first_is_Form = self._is_Form(args[0])
        first_is_Tensor = self._is_Tensor(args[0])
        first_is_DirichletBC = self._is_DirichletBC(args[0])
        first_is_Function = self._is_Function(args[0])
        assert first_is_Form or first_is_DirichletBC or first_is_Function or first_is_Tensor
        are_Form = list()
        are_Tensor = list()
        are_DirichletBC = list()
        are_Function = list()
        are_parametrized = list()
        for i in range(len(args)):
            if first_is_Form or first_is_Tensor:
                are_Form.append(self._is_Form(args[i]))
                are_Tensor.append(self._is_Tensor(args[i]))
                assert are_Form[i] or are_Tensor[i]
                if are_Form[i]:
                    if delay_assembly:
                        are_parametrized.append(True)
                    else:
                        are_parametrized.append(is_parametrized(args[i], form_iterator))
                else:
                    are_parametrized.append(False)
            elif first_is_DirichletBC:
                are_DirichletBC.append(self._is_DirichletBC(args[i]))
            elif first_is_Function:
                are_Function.append(self._is_Function(args[i]))
            else:
                return TypeError("Invalid input arguments to AffineExpansionStorage")
        # Actual init
        if first_is_Form or first_is_Tensor:
            any_is_parametrized = any(are_parametrized)
            if not any_is_parametrized:
                self._content = list()
                for (arg, arg_is_Form) in zip(args, are_Form):
                    if arg_is_Form:
                        self._content.append(assemble(arg, keep_diagonal=True))
                    else:
                        self._content.append(arg)
                self._type = "AssembledForm"
            else:
                self._content = list()
                for (arg, arg_is_Form, arg_is_parametrized) in zip(args, are_Form, are_parametrized):
                    if arg_is_Form and not arg_is_parametrized:
                        self._content.append(assemble(arg, keep_diagonal=True))
                    else:
                        self._content.append(arg) # either a Tensor or a parametrized Form
                self._type = "UnassembledForm"
        elif first_is_DirichletBC:
            assert all(are_DirichletBC)
            self._content = args
            self._type = "DirichletBC"
        elif first_is_Function:
            assert all(are_Function)
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
        
