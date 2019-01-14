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

from ufl import Form
from dolfin import assemble, DirichletBC
from rbnics.backends.abstract import AffineExpansionStorage as AbstractAffineExpansionStorage
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.function import Function
from rbnics.utils.decorators import backend_for, list_of, overload, tuple_of

# Generic backend
@backend_for("dolfin", inputs=((tuple_of(list_of(DirichletBC)), tuple_of(Form), tuple_of(Function.Type()), tuple_of(Matrix.Type()), tuple_of(Vector.Type()), tuple_of((Form, Matrix.Type())), tuple_of((Form, Vector.Type()))), ))
def AffineExpansionStorage(args):
    return _AffineExpansionStorage(args)

# Base implementation
class AffineExpansionStorage_Base(AbstractAffineExpansionStorage):
    def __init__(self, args):
        self._content = None
        
    def __getitem__(self, key):
        return self._content[key]
        
    def __iter__(self):
        return self._content.__iter__()
        
    def __len__(self):
        assert self._content is not None
        return len(self._content)
    
# Specialization for Dirichlet BCs
class AffineExpansionStorage_DirichletBC(AffineExpansionStorage_Base):
    def __init__(self, args):
        AffineExpansionStorage_Base.__init__(self, args)
        self._content = args
    
@overload
def _AffineExpansionStorage(args: tuple_of(list_of(DirichletBC))):
    return AffineExpansionStorage_DirichletBC(args)
    
# Specialization for forms
class AffineExpansionStorage_Form(AffineExpansionStorage_Base):
    def __init__(self, args):
        AffineExpansionStorage_Base.__init__(self, args)
        content = list()
        for arg in args:
            if isinstance(arg, Form):
                content.append(assemble(arg, keep_diagonal=True))
            elif isinstance(arg, (Matrix.Type(), Vector.Type())):
                content.append(arg)
            else:
                raise RuntimeError("Invalid argument to AffineExpansionStorage")
        self._content = tuple(content)

@overload
def _AffineExpansionStorage(args: (
    tuple_of(Form),
    tuple_of(Matrix.Type()),
    tuple_of(Vector.Type()),
    tuple_of((Form, Matrix.Type())),
    tuple_of((Form, Vector.Type()))
)):
    return AffineExpansionStorage_Form(args)
    
# Specialization for functions
class AffineExpansionStorage_Function(AffineExpansionStorage_Base):
    def __init__(self, args):
        AffineExpansionStorage_Base.__init__(self, args)
        self._content = args

@overload
def _AffineExpansionStorage(args: tuple_of(Function.Type())):
    return AffineExpansionStorage_Function(args)
