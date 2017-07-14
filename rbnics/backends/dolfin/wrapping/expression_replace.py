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

# This file contains a copy of UFL replace with a customized Replacer,
# which also handles Indexed and ListTensor cases

from ufl.algorithms.replace import replace as replace_for_terminals_only
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.corealg.multifunction import MultiFunction
from ufl.indexed import Indexed
from ufl.log import error
from ufl.tensors import ListTensor
from dolfin import Function, split

def expression_replace(expression, replacements):
    return replace(expression, replacements)

class Replacer(MultiFunction):
    def __init__(self, mapping):
        MultiFunction.__init__(self)
        for k in mapping:
            if (
                not k._ufl_is_terminal_
                    and
                not isinstance(k, (Indexed, ListTensor))
            ):
                error("This implementation can only replace Terminal objects or non terminal Indexed and ListTensor objects.")
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.iteritems()):
            error("Replacement expressions must have the same shape as what they replace.")
        # Create a mapping with terminals only
        self._mapping_terminals_only = dict()
        for (k, v) in mapping.iteritems():
            if k._ufl_is_terminal_:
                self._mapping_terminals_only[k] = v
        # Enrich the mapping extracting each Indexed from any ListTensor
        self._mapping = dict()
        for (k, v) in mapping.iteritems():
            self._mapping[k] = v
            if isinstance(k, ListTensor):
                assert all(isinstance(component, Indexed) for component in k.ufl_operands)
                assert all(
                  (len(component.ufl_operands) == 2 and isinstance(component.ufl_operands[0], Function) and isinstance(component.ufl_operands[1], MultiIndex))
                  for component in k.ufl_operands
                )
                assert all(
                  component.ufl_operands[0] == k.ufl_operands[-1].ufl_operands[0]
                  for component in k.ufl_operands
                )
                split_k = k.ufl_operands
                split_v = split(v)
                assert len(split_k) == len(split_v)
                for (sub_k, sub_v) in zip(split_k, split_v):
                    self._mapping[sub_k] = sub_v

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        e = self._mapping.get(o)
        if e is None:
            return o
        else:
            return e
            
    def indexed(self, o, Ap, ii):
        assert len(o.ufl_operands) == 2
        assert isinstance(o.ufl_operands[1], MultiIndex)
        if isinstance(o.ufl_operands[0], Function):
            indices = o.ufl_operands[1].indices()
            is_fixed = isinstance(indices[0], FixedIndex)
            assert all([isinstance(index, FixedIndex) == is_fixed for index in indices])
            is_mute = isinstance(indices[0], Index) # mute index for sum
            assert all([isinstance(index, Index) == is_mute for index in indices])
            assert (is_fixed and not is_mute) or (not is_fixed and is_mute)
            if is_fixed:
                assert o in self._mapping
                return self._mapping[o]
            elif is_mute:
                assert o.ufl_operands[0] in self._mapping
                return Indexed(self._mapping[o.ufl_operands[0]], o.ufl_operands[1])
            else:
                raise AssertionError("Invalid index")
        else:
            return replace_for_terminals_only(o, self._mapping_terminals_only)
    
    def list_tensor(self, o, *dops):
        if (
            all(isinstance(component, Indexed) for component in o.ufl_operands)
                and
            all(
              (len(component.ufl_operands) == 2 and isinstance(component.ufl_operands[0], Function) and isinstance(component.ufl_operands[1], MultiIndex))
              for component in o.ufl_operands
            )
                and
            all(
              component.ufl_operands[0] == o.ufl_operands[-1].ufl_operands[0]
              for component in o.ufl_operands
            )
        ):
            assert o in self._mapping
            return self._mapping[o]
        else:
            return replace_for_terminals_only(o, self._mapping_terminals_only)

    def coefficient_derivative(self, o):
        error("Derivatives should be applied before executing replace.")
        
def replace(e, mapping):
    """Replace objects in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.iteritems())
    
    # We have expanded derivative evaluation in ParametrizedTensorFactory
    assert not has_exact_type(e, CoefficientDerivative)
    
    return map_integrand_dags(Replacer(mapping2), e)
