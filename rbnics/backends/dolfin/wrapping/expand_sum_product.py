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

from sympy import expand_mul, IndexedBase, Mul, Pow, symbols
from ufl import Form
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import IndexSum, Product, Sum
from ufl.core.expr import Expr
from ufl.core.multiindex import MultiIndex
from ufl.corealg.multifunction import MultiFunction
from ufl.indexed import Indexed
from ufl.tensors import ComponentTensor
from rbnics.backends.dolfin.wrapping.remove_complex_nodes import remove_complex_nodes

def expand_sum_product(form):
    form = remove_complex_nodes(form) # TODO support forms in the complex field. This is currently needed otherwise conj((a+b)*c) does not get expanded.
    # Patch Expr.__mul__ and Expr.__rmul__
    patch_expr_mul()
    # Call sympy replacer
    expanded_form = map_integrand_dags(SympyExpander(), form)
    # Split sums
    expanded_split_form_integrals = list()
    for integral in expanded_form.integrals():
        expanded_split_form_integrands = list()
        split_sum(integral.integrand(), expanded_split_form_integrands)
        expanded_split_form_integrals.extend([integral.reconstruct(integrand=integrand) for integrand in expanded_split_form_integrands])
    expanded_split_form = Form(expanded_split_form_integrals)
    # Undo patch to Expr.__mul__ and Expr.__rmul__
    unpatch_expr_mul()
    # Return
    return expanded_split_form
    
class SympyExpander(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self.ufl_to_replaced_ufl = dict()
        self.ufl_to_sympy = dict()
        self.sympy_to_ufl = dict()
        self.ufl_to_sympy_id = dict()
        self.sympy_id_to_ufl = dict()
        
    expr = MultiFunction.reuse_if_untouched
    
    def operator(self, e, *ops):
        self._store_sympy_symbol(e)
        new_e = MultiFunction.reuse_if_untouched(self, e, *ops)
        if new_e != e:
            self._update_sympy_symbol(e, new_e)
        return new_e
        
    def terminal(self, o):
        self._store_sympy_symbol(o)
        return o
        
    def sum(self, e, arg1, arg2):
        if e not in self.ufl_to_replaced_ufl:
            self._store_sympy_symbol(e)
            def op(arg1, arg2):
                return arg1 + arg2
            (new_e, new_sympy_e) = self._apply_sympy_simplify(e, arg1, arg2, op)
            self._update_sympy_symbol(e, new_e, new_sympy_e)
            self.ufl_to_replaced_ufl[e] = new_e
            return new_e
        else:
            return self.ufl_to_replaced_ufl[e]

    def product(self, e, arg1, arg2):
        if e not in self.ufl_to_replaced_ufl:
            self._store_sympy_symbol(e)
            def op(arg1, arg2):
                return arg1*arg2
            (new_e, new_sympy_e) = self._apply_sympy_simplify(e, arg1, arg2, op)
            self._update_sympy_symbol(e, new_e, new_sympy_e)
            self.ufl_to_replaced_ufl[e] = new_e
            return new_e
        else:
            return self.ufl_to_replaced_ufl[e]
    
    def indexed(self, o, *dops):
        return self._transform_and_attach_multi_index(Indexed, o, *dops)
        
    def index_sum(self, o, *dops):
        return self._transform_and_attach_multi_index(IndexSum, o, *dops)
        
    def component_tensor(self, o, *dops):
        return self._transform_and_attach_multi_index(ComponentTensor, o, *dops)
    
    def _transform_and_attach_multi_index(self, Class, o, *dops):
        if o not in self.ufl_to_replaced_ufl:
            self._store_sympy_symbol(o)
            assert len(dops) == 2
            assert isinstance(dops[1], MultiIndex)
            transformed_ufl_operand_0 = list()
            split_sum(map_integrand_dags(self, dops[0]), transformed_ufl_operand_0)
            new_o = sum([Class(operand, dops[1]) for operand in transformed_ufl_operand_0])
            self._update_sympy_symbol(o, new_o)
            self.ufl_to_replaced_ufl[o] = new_o
            return new_o
        else:
            return self.ufl_to_replaced_ufl[o]
        
    def _apply_sympy_simplify(self, e, arg1, arg2, op):
        from rbnics.shape_parametrization.utils.symbolic import sympy_eval
        assert arg1 in self.ufl_to_sympy
        sympy_arg1 = self.ufl_to_sympy[arg1]
        assert arg2 in self.ufl_to_sympy
        sympy_arg2 = self.ufl_to_sympy[arg2]
        sympy_expanded_e = expand_mul(op(sympy_arg1, sympy_arg2))
        sympy_expanded_e = self._pow_to_mul(sympy_expanded_e) # Indexed does not support power with integer exponents, only multiplication
        ufl_expanded_e = sympy_eval(str(sympy_expanded_e), self.sympy_id_to_ufl)
        return (ufl_expanded_e, sympy_expanded_e)
        
    def _store_sympy_symbol(self, o):
        if isinstance(o, MultiIndex):
            pass
        else:
            if o not in self.ufl_to_sympy:
                sympy_id = "sympy" + str(len(self.ufl_to_sympy))
                if len(o.ufl_shape) == 0:
                    sympy_o = symbols(sympy_id)
                else:
                    sympy_o = IndexedBase(sympy_id, shape=o.ufl_shape)
                self.ufl_to_sympy[o] = sympy_o
                self.sympy_to_ufl[sympy_o] = o
                self.ufl_to_sympy_id[o] = sympy_id
                self.sympy_id_to_ufl[sympy_id] = o
                
    def _update_sympy_symbol(self, old_o, new_o, new_sympy_o=None):
        if new_sympy_o is not None:
            self.ufl_to_sympy[old_o] = new_sympy_o
            self.ufl_to_sympy[new_o] = new_sympy_o
            self.sympy_to_ufl[new_sympy_o] = new_o
        else:
            assert old_o in self.ufl_to_sympy
            sympy_o = self.ufl_to_sympy[old_o]
            self.ufl_to_sympy[new_o] = sympy_o
            self.sympy_to_ufl[sympy_o] = new_o
            assert old_o in self.ufl_to_sympy_id
            sympy_id = self.ufl_to_sympy_id[old_o]
            self.ufl_to_sympy_id[new_o] = sympy_id
            self.sympy_id_to_ufl[sympy_id] = new_o
        
    @staticmethod
    def _pow_to_mul(expr):
        """
        Convert integer powers in an expression to Muls, like a**2 => a*a.
        https://stackoverflow.com/questions/14264431/expanding-algebraic-powers-in-python-sympy
        """
        pows = [p for p in expr.atoms(Pow) if p.exp.is_Integer and p.exp >= 0]
        repl = dict(zip(pows, (Mul(*[p.base]*p.exp, evaluate=False) for p in pows)))
        output, _ = SympyExpander._non_eval_xreplace(expr, repl)
        return output
        
    @staticmethod
    def _non_eval_xreplace(expr, rule):
        """
        Duplicate of sympy's xreplace but with non-evaluate statement included.
        https://stackoverflow.com/questions/14264431/expanding-algebraic-powers-in-python-sympy
        """
        if expr in rule:
            return rule[expr], True
        elif rule:
            args = []
            changed = False
            for a in expr.args:
                a_xr = SympyExpander._non_eval_xreplace(a, rule)
                args.append(a_xr[0])
                changed |= a_xr[1]
            args = tuple(args)
            if changed:
                return expr.func(*args, evaluate=False), True
        return expr, False
        
def split_sum(input_, output):
    if isinstance(input_, Sum):
        for operand in input_.ufl_operands:
            split_sum(operand, output)
    elif isinstance(input_, IndexSum):
        assert len(input_.ufl_operands) == 2
        summand, indices = input_.ufl_operands
        assert isinstance(indices, MultiIndex)
        output_0 = list()
        split_sum(summand, output_0)
        output.extend([IndexSum(summand_0, indices) for summand_0 in output_0])
    else:
        output.append(input_)
        
# Sympy reconstructs expressions from scratch. If that expression contains a product, we do not want to go through
# the helper method _mult defined in ufl/exproperators.py, but rather create a plain UFL Product class.
Expr_mul = Expr.__mul__
Expr_rmul = Expr.__rmul__
def patch_expr_mul():
    def plain_mul(a, b):
        return Product(a, b)
    Expr.__mul__ = plain_mul
    Expr.__rmul__ = plain_mul
    
def unpatch_expr_mul():
    Expr.__mul__ = Expr_mul
    Expr.__rmul__ = Expr_rmul
