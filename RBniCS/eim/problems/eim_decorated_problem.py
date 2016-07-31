# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from itertools import product as cartesian_product
from dolfin import Function
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.eim.utils.io import AffineExpansionSeparatedFormsStorage
from RBniCS.eim.utils.ufl import SeparatedParametrizedForm
from RBniCS.eim.problems.eim_approximation import EIMApproximation

def EIMDecoratedProblem():
    @ProblemDecoratorFor(EIM)
    def EIMDecoratedProblem_Decorator(ParametrizedProblem_DerivedClass):
                
        @Extends(ParametrizedProblem_DerivedClass, preserve_class_name=True)
        class EIMDecoratedProblem_Class(ParametrizedProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for EIM reduced problems
                self.separated_forms = dict() # from terms to AffineExpansionSeparatedFormsStorage
                self.EIM_approximations = dict() # from coefficients to EIMApproximation
                
                # Preprocess each term in the affine expansions
                for term in self.terms:
                    forms = ParametrizedProblem_DerivedClass.assemble_operator(self, term)
                    Q = len(forms)
                    self.separated_forms[term] = AffineExpansionSeparatedFormsStorage(Q)
                    for q in range(Q):
                        self.separated_forms[term][q] = SeparatedParametrizedForm(forms[q])
                        self.separated_forms[term][q].separate()
                        # All parametrized coefficients should be approximated by EIM
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                if coeff not in self.EIM_approximations:
                                    self.EIM_approximations[coeff] = EIMApproximation(self.V, self, coeff, type(self).__name__ + "/eim/" + str(coeff.hash_code))
                
            ###########################     PROBLEM SPECIFIC     ########################### 
            ## @defgroup ProblemSpecific Problem specific methods
            #  @{
            
            @override
            def assemble_operator(self, term):
                if term in self.terms:
                    eim_forms = list()
                    for q in range(len(self.separated_forms[term])):
                        # Append forms computed with EIM, if applicable
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            eim_forms_coefficients_q_i = self.separated_forms[term][q].coefficients[i]
                            eim_forms_replacements_q_i__list = list()
                            for coeff in eim_forms_coefficients_q_i:
                                eim_forms_replacements_q_i__list.append(self.EIM_approximations[coeff].Z)
                            eim_forms_replacements_q_i__cartesian_product = cartesian_product(*eim_forms_replacements_q_i__list)
                            for t in eim_forms_replacements_q_i__cartesian_product:
                                new_coeffs = [Function(self.EIM_approximations[coeff].V, new_coeff) for new_coeff in t]
                                eim_forms.append(
                                    self.separated_forms[term][q].replace_placeholders(i, new_coeffs)
                                )
                        # Append forms which did not require EIM, if applicable
                        for unchanged_form in self.separated_forms[term][q]._form_unchanged:
                            eim_forms.append(unchanged_form)
                    return tuple(eim_forms)
                else:
                    return ParametrizedProblem_DerivedClass.assemble_operator(self, term) # may raise an exception
                    
            @override
            def compute_theta(self, term):
                original_thetas = ParametrizedProblem_DerivedClass.compute_theta(self, term) # may raise an exception
                if term in self.terms:
                    eim_thetas = list()
                    for q in range(len(original_thetas)):
                        # Append coefficients computed with EIM, if applicable
                        for i in range(len(self.separated_forms[term][q].coefficients)):
                            eim_thetas_q_i__list = list()
                            for coeff in self.separated_forms[term][q].coefficients[i]:
                                eim_thetas_q_i__list.append(self.EIM_approximations[coeff].compute_interpolated_theta())
                            eim_thetas_q_i__cartesian_product = cartesian_product(*eim_thetas_q_i__list)
                            for t in eim_thetas_q_i__cartesian_product:
                                eim_thetas_q_i_t = original_thetas[q]
                                for r in t:
                                    eim_thetas_q_i_t *= r
                                eim_thetas.append(eim_thetas_q_i_t)
                        # Append coefficients which did not require EIM, if applicable
                        for i in range(len(self.separated_forms[term][q]._form_unchanged)):
                            eim_thetas.append(original_thetas[q])
                    return tuple(eim_thetas)
                else:
                    return original_thetas
            #  @}
            ########################### end - PROBLEM SPECIFIC - end ########################### 
            
        # return value (a class) for the decorator
        return EIMDecoratedProblem_Class
        
    # return the decorator itself
    return EIMDecoratedProblem_Decorator
    
# For the sake of the user, since this is the only class that he/she needs to use, rename it to an easier name
EIM = EIMDecoratedProblem
