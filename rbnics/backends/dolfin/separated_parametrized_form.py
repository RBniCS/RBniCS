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

from logging import DEBUG, getLogger
from numpy import ones
from ufl import Argument, Form, Measure, replace
from ufl.algorithms import apply_transformer, expand_derivatives, Transformer
from ufl.algorithms.traversal import iter_expressions
from ufl.classes import FacetNormal
from ufl.core.multiindex import FixedIndex, Index as MuteIndex, MultiIndex
from ufl.corealg.traversal import pre_traversal, traverse_terminals
from ufl.geometry import GeometricQuantity
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.tensors import ComponentTensor, ListTensor
from dolfin import Constant
from dolfin.function.expression import BaseExpression
from rbnics.backends.abstract import SeparatedParametrizedForm as AbstractSeparatedParametrizedForm
from rbnics.backends.dolfin.wrapping import expand_sum_product, remove_complex_nodes, rewrite_quotients
from rbnics.utils.decorators import BackendFor, get_problem_from_solution, get_problem_from_solution_dot, ModuleWrapper

logger = getLogger("rbnics/backends/dolfin/separated_parametrized_form.py")

def BasicSeparatedParametrizedForm(backend, wrapping):
    class _BasicSeparatedParametrizedForm(AbstractSeparatedParametrizedForm):
        def __init__(self, form, strict=False):
            AbstractSeparatedParametrizedForm.__init__(self, form)
            form = expand_derivatives(form)
            form = expand_sum_product(form)
            form = rewrite_quotients(form)
            form = remove_complex_nodes(form) # TODO support forms in the complex field
            self._form = form
            self._coefficients = list() # of list of ParametrizedExpression
            self._placeholders = list() # of list of Constants
            self._placeholder_names = list() # of list of string
            self._form_with_placeholders = list() # of forms
            self._form_unchanged = list() # of forms
            # Internal usage
            self._NaN = float("NaN")
            # Strict mode when
            # * checking candidates to be added to coefficients which contain both parametrized and non parametrized leaves.
            # * checking for coefficients that are solution
            # If False (default)
            # * coefficient splitting is prevented, because separating the non parametrized part would result in more
            #   than one coefficient, and the candidate is accepted as the coefficient which contain both parametrized and non parametrized leaves.
            # * solutions are considered as parametrized
            # If True
            # * coefficient is split in order to assure that all coefficients only containt parametrized terms, at the expense of
            # * solutions and geometric quantities (except normals) are prevented for being collected in coefficients
            # a larger number of coefficients
            self._strict = strict
        
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
            
            logger.log(DEBUG, "***        SEPARATE FORM COEFFICIENTS        ***")
            
            logger.log(DEBUG, "1. Extract coefficients")
            integral_to_coefficients = dict()
            for integral in self._form.integrals():
                logger.log(DEBUG, "\t Currently on integrand " + str(integral.integrand()))
                self._coefficients.append(list()) # of ParametrizedExpression
                for e in iter_expressions(integral):
                    logger.log(DEBUG, "\t\t Expression " + str(e))
                    pre_traversal_e = [n for n in pre_traversal(e)]
                    tree_nodes_skip = [False for _ in pre_traversal_e]
                    for (n_i, n) in enumerate(pre_traversal_e):
                        if not tree_nodes_skip[n_i]:
                            # Skip expressions which are trivially non parametrized
                            if isinstance(n, Argument):
                                logger.log(DEBUG, "\t\t Node " + str(n) + " is skipped because it is an Argument")
                                continue
                            elif isinstance(n, Constant):
                                logger.log(DEBUG, "\t\t Node " + str(n) + " is skipped because it is a Constant")
                                continue
                            elif isinstance(n, MultiIndex):
                                logger.log(DEBUG, "\t\t Node " + str(n) + " is skipped because it is a MultiIndex")
                                continue
                            # Skip all expressions with at least one leaf which is an Argument
                            for t in traverse_terminals(n):
                                if isinstance(t, Argument):
                                    logger.log(DEBUG, "\t\t Node " + str(n) + " is skipped because it contains an Argument")
                                    break
                            else: # not broken
                                logger.log(DEBUG, "\t\t Node " + str(n) + " and its descendants are being analyzed for non-parametrized check")
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
                                            if isinstance(t, BaseExpression):
                                                if wrapping.is_pull_back_expression(t) and not wrapping.is_pull_back_expression_parametrized(t):
                                                    logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains a non-parametrized pulled back expression")
                                                    break
                                                else:
                                                    parameters = t._parameters
                                                    if "mu_0" not in parameters:
                                                        logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains a non-parametrized expression")
                                                        break
                                            elif isinstance(t, Constant):
                                                logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains a constant")
                                                break
                                            elif isinstance(t, GeometricQuantity) and not isinstance(t, FacetNormal) and self._strict:
                                                logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains a geometric quantity and strict mode is on")
                                                break
                                            elif wrapping.is_problem_solution_type(t):
                                                if not wrapping.is_problem_solution(t) and not wrapping.is_problem_solution_dot(t):
                                                    logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains a non-parametrized function")
                                                    break
                                                elif self._strict: # solutions are not allowed, break
                                                    if wrapping.is_problem_solution(t):
                                                        (_, _, solution) = wrapping.solution_identify_component(t)
                                                        logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains the solution of " + get_problem_from_solution(solution).name() + "and strict mode is on")
                                                        break
                                                    elif wrapping.is_problem_solution_dot(t):
                                                        (_, _, solution_dot) = wrapping.solution_dot_identify_component(t)
                                                        logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " causes the non-parametrized check to break because it contains the solution_dot of " + get_problem_from_solution_dot(solution_dot).name() + "and strict mode is on")
                                                    else:
                                                        raise RuntimeError("Unidentified solution found")
                                        else:
                                            at_least_one_expression_or_solution = False
                                            for t in traverse_terminals(d):
                                                if isinstance(t, BaseExpression): # which is parametrized, because previous for loop was not broken
                                                    at_least_one_expression_or_solution = True
                                                    logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " is a candidate after non-parametrized check because it contains the parametrized expression " + str(t))
                                                    break
                                                elif wrapping.is_problem_solution_type(t):
                                                    if wrapping.is_problem_solution(t):
                                                        at_least_one_expression_or_solution = True
                                                        (_, _, solution) = wrapping.solution_identify_component(t)
                                                        logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " is a candidate after non-parametrized check because it contains the solution of " + get_problem_from_solution(solution).name())
                                                        break
                                                    elif wrapping.is_problem_solution_dot(t):
                                                        at_least_one_expression_or_solution = True
                                                        (_, _, solution_dot) = wrapping.solution_dot_identify_component(t)
                                                        logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " is a candidate after non-parametrized check because it contains the solution_dot of " + get_problem_from_solution_dot(solution_dot).name())
                                                        break
                                            if at_least_one_expression_or_solution:
                                                all_candidates.append(d)
                                                pre_traversal_d = [q for q in pre_traversal(d)]
                                                for (q_i, q) in enumerate(pre_traversal_d):
                                                    assert q == pre_traversal_n[d_i + q_i] # make sure that we are marking the right node
                                                    internal_tree_nodes_skip[d_i + q_i] = True
                                            else:
                                                logger.log(DEBUG, "\t\t\t Descendant node " + str(d) + " has not passed the non-parametrized because it is not a parametrized expression or a solution")
                                # Evaluate candidates
                                if len(all_candidates) == 0: # the whole expression was actually non-parametrized
                                    logger.log(DEBUG, "\t\t Node " + str(n) + " is skipped because it is a non-parametrized coefficient")
                                    continue
                                elif len(all_candidates) == 1: # the whole expression was actually parametrized
                                    logger.log(DEBUG, "\t\t Node " + str(n) + " will be accepted because it is a non-parametrized coefficient")
                                    pass
                                else: # part of the expression was not parametrized, and separating the non parametrized part may result in more than one coefficient
                                    if self._strict: # non parametrized coefficients are not allowed, so split the expression
                                        logger.log(DEBUG, "\t\t\t Node " + str(n) + " will be accepted because it is a non-parametrized coefficient with more than one candidate. It will be split because strict mode is on. Its split coefficients are " + ", ".join([str(c) for c in all_candidates]))
                                    else: # non parametrized coefficients are allowed, so go on with the whole expression
                                        logger.log(DEBUG, "\t\t\t Node " + str(n) + " will be accepted because it is a non-parametrized coefficient with more than one candidate. It will not be split because strict mode is off. Splitting it would have resulted in more than one coefficient, namely " + ", ".join([str(c) for c in all_candidates]))
                                        all_candidates = [n]
                                # Add the coefficient(s)
                                for candidate in all_candidates:
                                    def preprocess_candidate(candidate):
                                        if isinstance(candidate, Indexed):
                                            assert len(candidate.ufl_operands) == 2
                                            assert isinstance(candidate.ufl_operands[1], MultiIndex)
                                            if all([isinstance(index, FixedIndex) for index in candidate.ufl_operands[1].indices()]):
                                                logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an Indexed expression with fixed indices, resulting in a candidate " + str(candidate) + " of type " + str(type(candidate)))
                                                return candidate # no further preprocessing needed
                                            else:
                                                logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an Indexed expression with at least one mute index, resulting in a candidate " + str(candidate.ufl_operands[0]) + " of type " + str(type(candidate.ufl_operands[0])))
                                                return preprocess_candidate(candidate.ufl_operands[0])
                                        elif isinstance(candidate, IndexSum):
                                            assert len(candidate.ufl_operands) == 2
                                            assert isinstance(candidate.ufl_operands[1], MultiIndex)
                                            assert all([isinstance(index, MuteIndex) for index in candidate.ufl_operands[1].indices()])
                                            logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an IndexSum expression, resulting in a candidate " + str(candidate.ufl_operands[0]) + " of type " + str(type(candidate.ufl_operands[0])))
                                            return preprocess_candidate(candidate.ufl_operands[0])
                                        elif isinstance(candidate, ListTensor):
                                            candidates = set([preprocess_candidate(component) for component in candidate.ufl_operands])
                                            if len(candidates) == 1:
                                                preprocessed_candidate = candidates.pop()
                                                logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an ListTensor expression with a unique preprocessed component, resulting in a candidate " + str(preprocessed_candidate) + " of type " + str(type(preprocessed_candidate)))
                                                return preprocess_candidate(preprocessed_candidate)
                                            else:
                                                at_least_one_mute_index = False
                                                candidates_from_components = list()
                                                for component in candidates:
                                                    assert isinstance(component, (ComponentTensor, Indexed))
                                                    assert len(component.ufl_operands) == 2
                                                    assert isinstance(component.ufl_operands[1], MultiIndex)
                                                    if not all([isinstance(index, FixedIndex) for index in component.ufl_operands[1].indices()]):
                                                        at_least_one_mute_index = True
                                                    candidates_from_components.append(preprocess_candidate(component.ufl_operands[0]))
                                                if at_least_one_mute_index:
                                                    candidates_from_components = set(candidates_from_components)
                                                    assert len(candidates_from_components) == 1
                                                    preprocessed_candidate = candidates_from_components.pop()
                                                    logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an ListTensor expression with multiple preprocessed components with at least one mute index, resulting in a candidate " + str(preprocessed_candidate) + " of type " + str(type(preprocessed_candidate)))
                                                    return preprocess_candidate(preprocessed_candidate)
                                                else:
                                                    logger.log(DEBUG, "\t\t\t Preprocessed descendant node " + str(candidate) + " as an ListTensor expression with multiple preprocessed components with fixed indices, resulting in a candidate " + str(candidate) + " of type " + str(type(candidate)))
                                                    return candidate # no further preprocessing needed
                                        else:
                                            logger.log(DEBUG, "\t\t\t No preprocessing required for descendant node " + str(candidate) + " as a coefficient of type " + str(type(candidate)))
                                            return candidate
                                    preprocessed_candidate = preprocess_candidate(candidate)
                                    if preprocessed_candidate not in self._coefficients[-1]:
                                        self._coefficients[-1].append(preprocessed_candidate)
                                    logger.log(DEBUG, "\t\t\t Accepting descendant node " + str(preprocessed_candidate) + " as a coefficient of type " + str(type(preprocessed_candidate)))
                        else:
                            logger.log(DEBUG, "\t\t Node " + str(n) + " to be skipped because it is a descendant of a coefficient which has already been detected")
                if len(self._coefficients[-1]) == 0: # then there were no coefficients to extract
                    logger.log(DEBUG, "\t There were no coefficients to extract")
                    self._coefficients.pop() # remove the (empty) element that was added to possibly store coefficients
                else:
                    logger.log(DEBUG, "\t Extracted coefficients are:")
                    for c in self._coefficients[-1]:
                        logger.log(DEBUG, "\t\t" + str(c))
                    integral_to_coefficients[integral] = self._coefficients[-1]
            
            logger.log(DEBUG, "2. Prepare placeholders and forms with placeholders")
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
                    logger.log(DEBUG, "\t Adding form for integrand " + str(integral.integrand()) + " to unchanged forms")
                    self._form_unchanged.append(integral.integrand()*measure)
                else:
                    logger.log(DEBUG, "\t Preparing form with placeholders for integrand " + str(integral.integrand()))
                    self._placeholders.append(list()) # of Constants
                    placeholders_dict = dict()
                    for c in integral_to_coefficients[integral]:
                        self._placeholders[-1].append(Constant(self._NaN*ones(c.ufl_shape)))
                        placeholders_dict[c] = self._placeholders[-1][-1]
                        logger.log(DEBUG, "\t\t " + str(placeholders_dict[c]) + " is the placeholder for " + str(c))
                    replacer = _SeparatedParametrizedForm_Replacer(placeholders_dict)
                    new_integrand = apply_transformer(integral.integrand(), replacer)
                    self._form_with_placeholders.append(new_integrand*measure)
                
            logger.log(DEBUG, "3. Assert that there are no parametrized expressions left")
            for form in self._form_with_placeholders:
                for integral in form.integrals():
                    for e in pre_traversal(integral.integrand()):
                        if isinstance(e, BaseExpression):
                            assert not (wrapping.is_pull_back_expression(e) and wrapping.is_pull_back_expression_parametrized(e)), "Form " + str(integral) + " still contains a parametrized pull back expression"
                            parameters = e._parameters
                            assert "mu_0" not in parameters, "Form " + str(integral) + " still contains a parametrized expression"
            
            logger.log(DEBUG, "4. Prepare coefficients hash codes")
            for addend in self._coefficients:
                self._placeholder_names.append(list()) # of string
                for factor in addend:
                    self._placeholder_names[-1].append(wrapping.expression_name(factor))
                    
            logger.log(DEBUG, "5. Assert list length consistency")
            assert len(self._coefficients) == len(self._placeholders)
            assert len(self._coefficients) == len(self._placeholder_names)
            for (c, p, pn) in zip(self._coefficients, self._placeholders, self._placeholder_names):
                assert len(c) == len(p)
                assert len(c) == len(pn)
            assert len(self._coefficients) == len(self._form_with_placeholders)
            
            logger.log(DEBUG, "*** DONE - SEPARATE FORM COEFFICIENTS - DONE ***")
            logger.log(DEBUG, "")

        @property
        def coefficients(self):
            return self._coefficients
            
        @property
        def unchanged_forms(self):
            return self._form_unchanged

        def replace_placeholders(self, i, new_coefficients):
            assert len(new_coefficients) == len(self._placeholders[i])
            replacements = dict((placeholder, new_coefficient) for (placeholder, new_coefficient) in zip(self._placeholders[i], new_coefficients))
            return replace(self._form_with_placeholders[i], replacements)
            
        def placeholders_names(self, i):
            return self._placeholder_names[i]
    
    return _BasicSeparatedParametrizedForm

from rbnics.backends.dolfin.wrapping import expression_name, is_problem_solution, is_problem_solution_dot, is_problem_solution_type, is_pull_back_expression, is_pull_back_expression_parametrized, solution_dot_identify_component, solution_identify_component
backend = ModuleWrapper()
wrapping = ModuleWrapper(is_problem_solution, is_problem_solution_dot, is_problem_solution_type, is_pull_back_expression, is_pull_back_expression_parametrized, solution_dot_identify_component, solution_identify_component, expression_name=expression_name)
SeparatedParametrizedForm_Base = BasicSeparatedParametrizedForm(backend, wrapping)

@BackendFor("dolfin", inputs=(Form, ))
class SeparatedParametrizedForm(SeparatedParametrizedForm_Base):
    pass
