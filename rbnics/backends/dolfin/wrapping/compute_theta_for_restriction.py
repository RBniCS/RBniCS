# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def compute_theta_for_restriction(restricted_term_to_original_term):

    def compute_theta_for_restriction_decorator(compute_theta):

        def compute_theta_for_restriction_decorator_impl(self, term):
            original_term = restricted_term_to_original_term.get(term)
            if original_term is None:  # term was not a restricted term
                return compute_theta(self, term)
            else:
                assert term.endswith("_restricted")
                return compute_theta(self, original_term)

        return compute_theta_for_restriction_decorator_impl

    return compute_theta_for_restriction_decorator
