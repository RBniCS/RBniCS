# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from numbers import Number
from dolfin import Expression
from rbnics.backends.dolfin.wrapping.parametrized_constant import (
    is_parametrized_constant, parametrized_constant_to_float)
from rbnics.utils.cache import Cache
from rbnics.utils.decorators.sync_setters import _original_setters
from rbnics.utils.test import AttachInstanceMethod, PatchInstanceMethod


# This ideally should be a subclass of Expression. However, dolfin manual
# states that subclassing Expression may be significantly slower than using
# JIT-compiled expressions. To this end we avoid subclassing expression and
# just add the set_mu method by patching the expression instance
def ParametrizedExpression(truth_problem, parametrized_expression_code=None, *args, **kwargs):
    if parametrized_expression_code is None:
        return None

    assert "mu" in kwargs
    mu = kwargs["mu"]
    assert mu is not None
    assert isinstance(mu, tuple)
    P = len(mu)
    for p in range(P):
        assert isinstance(parametrized_expression_code, (tuple, str))
        if isinstance(parametrized_expression_code, tuple):
            if isinstance(parametrized_expression_code[0], tuple):
                matrix_after_replacements = list()
                for row in parametrized_expression_code:
                    assert isinstance(row, tuple)
                    new_row = list()
                    for item in row:
                        assert isinstance(item, str)
                        new_row.append(item.replace("mu[" + str(p) + "]", "mu_" + str(p)))
                    new_row = tuple(new_row)
                    matrix_after_replacements.append(new_row)
                parametrized_expression_code = tuple(matrix_after_replacements)
            else:
                vector_after_replacements = list()
                for item in parametrized_expression_code:
                    assert isinstance(item, str)
                    vector_after_replacements.append(item.replace("mu[" + str(p) + "]", "mu_" + str(p)))
                parametrized_expression_code = tuple(vector_after_replacements)
        elif isinstance(parametrized_expression_code, str):
            parametrized_expression_code = parametrized_expression_code.replace("mu[" + str(p) + "]", "mu_" + str(p))
        else:
            raise TypeError("Invalid expression type in ParametrizedExpression")

    # Detect mesh
    if "domain" in kwargs:
        mesh = kwargs["domain"]
    else:
        mesh = truth_problem.V.mesh()

    # Prepare a dictionary of mu
    mu_dict = dict()
    for (p, mu_p) in enumerate(mu):
        assert isinstance(mu_p, (Expression, Number))
        if isinstance(mu_p, Number):
            mu_dict["mu_" + str(p)] = mu_p
        elif isinstance(mu_p, Expression):
            assert is_parametrized_constant(mu_p)
            mu_dict["mu_" + str(p)] = parametrized_constant_to_float(mu_p, point=mesh.coordinates()[0])
    del kwargs["mu"]
    kwargs.update(mu_dict)

    # Initialize expression
    expression = Expression(parametrized_expression_code, *args, **kwargs)
    expression._mu = mu  # to avoid repeated assignments

    # Store mesh
    expression._mesh = mesh

    # Cache all problem -> expression relation
    first_parametrized_expression_for_truth_problem = (truth_problem not in _truth_problem_to_parametrized_expressions)
    if first_parametrized_expression_for_truth_problem:
        _truth_problem_to_parametrized_expressions[truth_problem] = list()
    _truth_problem_to_parametrized_expressions[truth_problem].append(expression)

    # Keep mu in sync
    if first_parametrized_expression_for_truth_problem:

        def generate_overridden_set_mu(standard_set_mu):

            def overridden_set_mu(self, mu):
                standard_set_mu(mu)
                for expression_ in _truth_problem_to_parametrized_expressions[self]:
                    if expression_._mu is not mu:
                        expression_._set_mu(mu)

            return overridden_set_mu

        if ("set_mu" in _original_setters
                and truth_problem in _original_setters["set_mu"]):
            # truth_problem.set_mu was already patched by the decorator @sync_setters
            standard_set_mu = _original_setters["set_mu"][truth_problem]
            overridden_set_mu = generate_overridden_set_mu(standard_set_mu)
            _original_setters["set_mu"][truth_problem] = types.MethodType(overridden_set_mu, truth_problem)
        else:
            standard_set_mu = truth_problem.set_mu
            overridden_set_mu = generate_overridden_set_mu(standard_set_mu)
            PatchInstanceMethod(truth_problem, "set_mu", overridden_set_mu).patch()

    def expression_set_mu(self, mu):
        assert isinstance(mu, tuple)
        assert len(mu) >= len(self._mu)
        mu = mu[:len(self._mu)]
        for (p, mu_p) in enumerate(mu):
            assert isinstance(mu_p, (Expression, Number))
            if isinstance(mu_p, Number):
                setattr(self, "mu_" + str(p), mu_p)
            elif isinstance(mu_p, Expression):
                assert is_parametrized_constant(mu_p)
                setattr(self, "mu_" + str(p), parametrized_constant_to_float(mu_p, point=mesh.coordinates()[0]))
        self._mu = mu

    AttachInstanceMethod(expression, "_set_mu", expression_set_mu).attach()
    # Note that this override is different from the one that we use in decorated problems,
    # since (1) we do not want to define a new child class, (2) we have to execute some preprocessing
    # on the data, (3) it is a one-way propagation rather than a sync.
    # For these reasons, the decorator @sync_setters is not used but we partially duplicate some code

    # Possibly also keep time in sync
    if hasattr(truth_problem, "set_time"):
        if first_parametrized_expression_for_truth_problem:

            def generate_overridden_set_time(standard_set_time):

                def overridden_set_time(self, t):
                    standard_set_time(t)
                    for expression_ in _truth_problem_to_parametrized_expressions[self]:
                        if hasattr(expression_, "t"):
                            if expression_.t != t:
                                assert isinstance(expression_.t, Number)
                                expression_.t = t

                return overridden_set_time

            if ("set_time" in _original_setters
                    and truth_problem in _original_setters["set_time"]):
                # truth_problem.set_time was already patched by the decorator @sync_setters
                standard_set_time = _original_setters["set_time"][truth_problem]
                overridden_set_time = generate_overridden_set_time(standard_set_time)
                _original_setters["set_time"][truth_problem] = types.MethodType(overridden_set_time, truth_problem)
            else:
                standard_set_time = truth_problem.set_time
                overridden_set_time = generate_overridden_set_time(standard_set_time)
                PatchInstanceMethod(truth_problem, "set_time", overridden_set_time).patch()

    return expression


_truth_problem_to_parametrized_expressions = Cache()
