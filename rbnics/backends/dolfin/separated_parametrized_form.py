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

from numpy import ones, zeros
from dolfin import Constant, Expression, Function, log, PROGRESS
from ufl import Argument, Form, Measure, replace
from ufl.algebra import Sum
from ufl.algorithms import expand_derivatives, Transformer
from ufl.algorithms.traversal import iter_expressions
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.corealg.traversal import pre_traversal, traverse_terminals
from ufl.indexed import Indexed
from ufl.tensors import ComponentTensor, ListTensor
from rbnics.utils.io import ExportableList
from rbnics.utils.decorators import BackendFor, Extends, override
from rbnics.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm
from rbnics.backends.dolfin.wrapping import expression_name

@Extends(AbstractSeparatedParametrizedForm)
@BackendFor("dolfin", inputs=(Form, ))
class SeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
    def __init__(self, form):
        AbstractSeparatedParametrizedForm.__init__(self, form)
        form = expand_derivatives(form)
        self._form = form
        self._coefficients = list() # of list of ParametrizedExpression
        self._placeholders = list() # of list of Constants
        self._placeholder_names = list() # of list of string
        self._form_with_placeholders = list() # of forms
        self._form_unchanged = list() # of forms
        # Internal usage
        self._NaN = float('NaN')
    
    @override
    def separate(self):
        class _SeparatedParametrizedForm_Replacer(Transformer):
            def __init__(self, mapping):
                Transformer.__init__(self)
                self.mapping = mapping

            def operator(self, e, *ops):
                if e in self.mapping:
                    return self.mapping[e]
                else:
                    return e._ufl_expr_reconstruct_(*ops)
                
            def terminal(self, e):
                return self.mapping.get(e, e)
        
        log(PROGRESS, "***        SEPARATE FORM COEFFICIENTS        ***")
        
        log(PROGRESS, "1. Extract coefficients")
        integral_to_coefficients = dict()
        for integral in self._form.integrals():
            log(PROGRESS, "\t Currently on integrand " + str(integral.integrand()))
            assert not isinstance(integral.integrand(), Sum), "Please write your form as a*u*v*dx + b*u*v*dx rather than (a*u*v + b*u*v)*dx, otherwise skipping tree nodes may not work."
            self._coefficients.append( list() ) # of ParametrizedExpression
            for e in iter_expressions(integral):
                log(PROGRESS, "\t\t Expression " + str(e))
                pre_traversal_e = [n for n in pre_traversal(e)]
                tree_nodes_skip = [False for _ in pre_traversal_e]
                for (n_i, n) in enumerate(pre_traversal_e):
                    if not tree_nodes_skip[n_i]:
                        # Skip expressions which are an Argument or (only a) multiindex
                        if isinstance(n, Argument):
                            log(PROGRESS, "\t\t Node " + str(n) + " is skipped because it is an Argument")
                            continue
                        elif isinstance(n, MultiIndex):
                            log(PROGRESS, "\t\t Node " + str(n) + " is skipped because it is a MultiIndex")
                            continue
                        if isinstance(n, Constant):
                            log(PROGRESS, "\t\t Node " + str(n) + " is skipped because it is a Constant")
                            continue
                        # Skip all expressions with at least one leaf which is an Argument
                        for t in traverse_terminals(n):
                            if isinstance(t, Argument):
                                log(PROGRESS, "\t\t Node " + str(n) + " is skipped because it contains an Argument")
                                break
                        else: # not broken
                            log(PROGRESS, "\t\t Node " + str(n) + " and its descendants are being analyzed for non-parametrized check")
                            # Make sure to skip all descendants of this node in the outer loop
                            # Note that a map with key set to the expression is not enough to 
                            # mark the node as visited, since the same expression may appear
                            # on different sides of the tree
                            pre_traversal_n = [d for d in pre_traversal(n)]
                            for (d_i, d) in enumerate(pre_traversal_n):
                                assert d == pre_traversal_e[n_i + d_i] # make sure that we are marking the right node
                                tree_nodes_skip[n_i + d_i] = True
                            # We might be able to strip any (non-parametrized) expression out
                            all_candidates = list()
                            internal_tree_nodes_skip = [False for _ in pre_traversal_n]
                            for (d_i, d) in enumerate(pre_traversal_n):
                                if not internal_tree_nodes_skip[d_i]:
                                    # Skip all expressions where at least one leaf is not parametrized
                                    for t in traverse_terminals(d):
                                        if (isinstance(t, Expression) and "mu_0" not in t.user_parameters) or isinstance(t, Constant):
                                            log(PROGRESS, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it is a non-parametrized expression")
                                            break
                                    else:
                                        at_least_one_expression_or_function = False
                                        for t in traverse_terminals(d):
                                            if isinstance(t, (Expression, Function)): # Functions are always assumed to be parametrized
                                                at_least_one_expression_or_function = True
                                        if at_least_one_expression_or_function:
                                            log(PROGRESS, "\t\t\t Descendant node " + str(d) + " is a candidate after non-parametrized check")
                                            all_candidates.append(d)
                                            pre_traversal_d = [q for q in pre_traversal(d)]
                                            for (q_i, q) in enumerate(pre_traversal_d):
                                                assert q == pre_traversal_n[d_i + q_i] # make sure that we are marking the right node
                                                internal_tree_nodes_skip[d_i + q_i] = True
                                        else:
                                            log(PROGRESS, "\t\t\t Descendant node " + str(d) + " has not passed the non-parametrized because is not an expression")
                            # Evaluate candidates
                            if len(all_candidates) == 0: # the whole expression was actually non-parametrized
                                log(PROGRESS, "\t\t Node " + str(n) + " is skipped because is a non-parametrized coefficient")
                                continue
                            elif len(all_candidates) == 1: # the whole expression was actually parametrized
                                candidate = all_candidates[0]
                            else: # part of the expression was not parametrized, but separating the non parametrized part would result in more than one coefficient
                                candidate = n
                                log(PROGRESS, "\t\t\t Node " + str(n) + " was not split because it would have resulted in more than one coefficient, namely " + ", ".join([str(c) for c in all_candidates]))
                            # Add the coefficient
                            if isinstance(candidate, Indexed):
                                assert isinstance(candidate.ufl_operands[1], MultiIndex)
                                assert len(candidate.ufl_operands) == 2
                                indices = candidate.ufl_operands[1].indices()
                                is_fixed = isinstance(indices[0], FixedIndex)
                                assert all([isinstance(index, FixedIndex) == is_fixed for index in indices])
                                is_mute = isinstance(indices[0], Index) # mute index for sum
                                assert all([isinstance(index, Index) == is_mute for index in indices])
                                assert (is_fixed and not is_mute) or (not is_fixed and is_mute)
                                if is_fixed:
                                    self._coefficients[-1].append(candidate)
                                    log(PROGRESS, "\t\t\t Accepting descandant node " + str(candidate) + " as an Indexed expression with fixed index, resulting in a coefficient " + str(candidate.ufl_operands[0]) + " of type " + str(type(candidate.ufl_operands[0])) + " for fixed index " + str(candidate.ufl_operands[1]))
                                elif is_mute:
                                    self._coefficients[-1].append(candidate.ufl_operands[0])
                                    log(PROGRESS, "\t\t\t Accepting descandant node " + str(candidate) + " as an Indexed expression with mute index, resulting in a coefficient " + str(candidate.ufl_operands[0]) + " of type " + str(type(candidate.ufl_operands[0])))
                                else:
                                    raise AssertionError("Invalid index")
                            else:
                                assert not isinstance(candidate, (ListTensor, ComponentTensor))
                                self._coefficients[-1].append(candidate)
                                log(PROGRESS, "\t\t\t Accepting descandant node " + str(candidate) + " as a coefficient of type " + str(type(candidate)))
                    else:
                        log(PROGRESS, "\t\t Node " + str(n) + " to be skipped because is a descendant of a coefficient which has already been detected")
            if len(self._coefficients[-1]) == 0: # then there were no coefficients to extract
                log(PROGRESS, "\t There were no coefficients to extract")
                self._coefficients.pop() # remove the (empty) element that was added to possibly store coefficients
            else:
                log(PROGRESS, "\t Extracted coefficients are:\n\t\t" + "\n\t\t".join([str(c) for c in self._coefficients[-1]]))
                integral_to_coefficients[integral] = self._coefficients[-1]
        
        log(PROGRESS, "2. Prepare placeholders and forms with placeholders")
        for integral in self._form.integrals():
            # Prepare measure for the new form (from firedrake/mg/ufl_utils.py)
            measure = Measure(
                integral.integral_type(),
                domain=integral.ufl_domain(),
                subdomain_id=integral.subdomain_id(),
                subdomain_data=integral.subdomain_data(),
                metadata=integral.metadata()
            )
            if integral not in integral_to_coefficients:
                log(PROGRESS, "\t Adding form for integrand " + str(integral.integrand()) + " to unchanged forms")
                self._form_unchanged.append(integral.integrand()*measure)
            else:
                log(PROGRESS, "\t Preparing form with placeholders for integrand " + str(integral.integrand()))
                self._placeholders.append( list() ) # of Constants
                placeholders_dict = dict()
                for c in integral_to_coefficients[integral]:
                    self._placeholders[-1].append( Constant(self._NaN*ones(c.ufl_shape)) )
                    placeholders_dict[c] = self._placeholders[-1][-1]
                replacer = _SeparatedParametrizedForm_Replacer(placeholders_dict)
                new_integrand = replacer.visit(integral.integrand())
                self._form_with_placeholders.append(new_integrand*measure)
            
        log(PROGRESS, "3. Assert that there are no parametrized expressions left")
        for form in self._form_with_placeholders:
            for integral in form.integrals():
                for e in pre_traversal(integral.integrand()):
                    assert not (isinstance(e, Expression) and "mu_0" in e.user_parameters), "Form " + str(i) + " still contains a parametrized expression"
        
        log(PROGRESS, "4. Prepare coefficients hash codes")
        for addend in self._coefficients:
            self._placeholder_names.append( list() ) # of string
            for factor in addend:
                self._placeholder_names[-1].append(expression_name(factor))
                
        log(PROGRESS, "5. Assert list length consistency")
        assert len(self._coefficients) == len(self._placeholders)
        assert len(self._coefficients) == len(self._placeholder_names)
        for (c, p, pn) in zip(self._coefficients, self._placeholders, self._placeholder_names):
            assert len(c) == len(p)
            assert len(c) == len(pn)
        assert len(self._coefficients) == len(self._form_with_placeholders)
        
        log(PROGRESS, "*** DONE - SEPARATE FORM COEFFICIENTS - DONE ***")
        log(PROGRESS, "")

    @override        
    @property
    def coefficients(self):
        return self._coefficients
        
    @override
    @property
    def unchanged_forms(self):
        return self._form_unchanged

    @override        
    def replace_placeholders(self, i, new_coefficients):
        assert len(new_coefficients) == len(self._placeholders[i])
        replacements = dict((placeholder, new_coefficient) for (placeholder, new_coefficient) in zip(self._placeholders[i], new_coefficients))
        return replace(self._form_with_placeholders[i], replacements)
        
    @override
    def placeholders_names(self, i):
        return self._placeholder_names[i]

