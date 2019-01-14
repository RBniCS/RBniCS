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

def compute_theta_for_restriction(restricted_term_to_original_term):
    def compute_theta_for_restriction_decorator(compute_theta):
        def compute_theta_for_restriction_decorator_impl(self, term):
            original_term = restricted_term_to_original_term.get(term)
            if original_term is None: # term was not a restricted term
                return compute_theta(self, term)
            else:
                assert term.endswith("_restricted")
                return compute_theta(self, original_term)
            
        return compute_theta_for_restriction_decorator_impl
    return compute_theta_for_restriction_decorator
