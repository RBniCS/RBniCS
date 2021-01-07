# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import zeros
from dolfin import Constant, FunctionSpace, split, TestFunction, TrialFunction
from rbnics.backends.dolfin.wrapping.expression_replace import replace
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC
from rbnics.utils.decorators import overload


def assemble_operator_for_restriction(restricted_term_to_original_term, test=None, trial=None):

    def assemble_operator_for_restriction_decorator(assemble_operator):

        def assemble_operator_for_restriction_decorator_impl(self, term):
            original_term = restricted_term_to_original_term.get(term)
            if original_term is None:  # term was not a original term
                return assemble_operator(self, term)
            else:
                assert (term.endswith("_restricted") or term.startswith("inner_product_")
                        or term.startswith("dirichlet_bc_"))
                if term.endswith("_restricted") or term.startswith("inner_product_"):
                    test_int = _to_int(self.V, test)
                    trial_int = _to_int(self.V, trial)
                    assert test_int is not None or trial_int is not None
                    replacements = dict()
                    if test_int is not None:
                        original_test = split(TestFunction(self.V))
                        original_test = original_test[test_int]
                        restricted_test = TestFunction(self.V.sub(test_int).collapse())
                        replacements[original_test] = restricted_test
                    if trial_int is not None:
                        original_trial = split(TrialFunction(self.V))
                        original_trial = original_trial[trial_int]
                        restricted_trial = TrialFunction(self.V.sub(trial_int).collapse())
                        replacements[original_trial] = restricted_trial
                    return tuple(replace(op, replacements) for op in assemble_operator(self, original_term))
                elif term.startswith("dirichlet_bc_"):
                    assert test is None
                    trial_int = _to_int(self.V, trial)
                    assert trial_int is not None
                    original_dirichlet_bcs = assemble_operator(self, original_term)
                    restricted_dirichlet_bcs = list()
                    for original_dirichlet_bc in original_dirichlet_bcs:
                        restricted_dirichlet_bc = list()
                        for original_dirichlet_bc_i in original_dirichlet_bc:
                            V = original_dirichlet_bc_i.function_space()
                            parent_V = V
                            assert hasattr(parent_V, "_root_space_after_sub")
                            while parent_V._root_space_after_sub is not None:
                                parent_V = parent_V._root_space_after_sub
                                assert hasattr(parent_V, "_root_space_after_sub")
                            V_component = [int(c) for c in V.component()]
                            assert len(V_component) >= 1
                            assert V_component[0] == trial_int
                            restricted_V = parent_V.sub(V_component[0]).collapse()
                            for c in V_component[1:]:
                                restricted_V = restricted_V.sub(c)
                            restricted_value = Constant(zeros(original_dirichlet_bc_i.value().ufl_shape))
                            args = list()
                            args.append(restricted_V)
                            args.append(restricted_value)
                            args.extend(original_dirichlet_bc_i._domain)
                            kwargs = original_dirichlet_bc_i._kwargs
                            restricted_dirichlet_bc.append(DirichletBC(*args, **kwargs))
                        assert len(restricted_dirichlet_bc) == len(original_dirichlet_bc)
                        restricted_dirichlet_bcs.append(restricted_dirichlet_bc)
                    assert len(restricted_dirichlet_bcs) == len(original_dirichlet_bcs)
                    return tuple(restricted_dirichlet_bcs)

        return assemble_operator_for_restriction_decorator_impl

    return assemble_operator_for_restriction_decorator


@overload(FunctionSpace, (int, None))
def _to_int(V, restrict_to):
    return restrict_to


@overload(FunctionSpace, str)
def _to_int(V, restrict_to):
    assert hasattr(V, "_component_to_index")
    return V._component_to_index[restrict_to]
