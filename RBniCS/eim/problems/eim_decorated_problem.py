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
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.eim.utils.io import AffineExpansionSeparatedFormsStorage
from RBniCS.eim.utils.ufl import SeparatedParametrizedForm
from RBniCS.eim.problems.eim_approximation import EIMApproximation

def EIMDecoratedProblem(**decorator_kwargs):
    from RBniCS.eim.problems.exact_parametrized_functions_decorated_problem import ExactParametrizedFunctions
    
    @ProblemDecoratorFor(EIM, ExactAlgorithm=ExactParametrizedFunctions)
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
                        for addend in self.separated_forms[term][q].coefficients:
                            for factor in addend:
                                if factor not in self.EIM_approximations:
                                    self.EIM_approximations[coeff] = EIMApproximation(self, InterpolationInput(factor, self.V), type(self).__name__ + "/eim/" + str(factor.hash_code))
                                    
                # Avoid useless assignments
                self._update_N_EIM_in_compute_theta.__func__.previous_kwargs = None
                
            @override
            def solve(self, **kwargs):
                self._update_N_EIM_in_compute_theta(**kwargs)
                return ParametrizedProblem_DerivedClass.solve(self, **kwargs)
            
            def _update_N_EIM_in_compute_theta(self, **kwargs):
                if kwargs != self._update_N_EIM_in_compute_theta.__func__.previous_kwargs:
                    if "EIM" in kwargs:
                        self.compute_theta.__func__.N_EIM = dict()
                        N_EIM = kwargs["EIM"]
                        for term in self.separated_forms:
                            self.compute_theta.__func__.N_EIM[term] = list()
                            if isinstance(N_EIM, dict):
                                assert term in N_EIM
                                assert len(N_EIM[term]) == len(self.separated_forms[term])
                                for N_eim_term_q in N_EIM[term]:
                                    self.compute_theta.__func__.N_EIM[term].append(N_eim_term_q)
                            else:
                                assert isinstance(N_EIM, int)
                                for _ in self.separated_forms[term]:
                                    self.compute_theta.__func__.N_EIM[term].append(N_EIM)
                    else:
                        if hasattr(self.compute_theta.__func__, "N_EIM"):
                            delattr(self.compute_theta.__func__, "N_EIM")
                    self._update_N_EIM_in_compute_theta.__func__.previous_kwargs = kwargs
                
                
            ###########################     PROBLEM SPECIFIC     ########################### 
            ## @defgroup ProblemSpecific Problem specific methods
            #  @{
            
            @override
            def assemble_operator(self, term):
                if term in self.terms:
                    eim_forms = list()
                    for form in self.separated_forms[term]:
                        # Append forms computed with EIM, if applicable
                        for (index, addend) in form.coefficients:
                            replacements__list = list()
                            for factor in addend:
                                replacements__list.append(self.EIM_approximations[factor].Z)
                            replacements__cartesian_product = cartesian_product(*replacements__list)
                            for new_coeffs in replacements__cartesian_product:
                                eim_forms.append(
                                    form.replace_placeholders(index, new_coeffs)
                                )
                        # Append forms which did not require EIM, if applicable
                        for unchanged_form in form._form_unchanged:
                            eim_forms.append(unchanged_form)
                    return tuple(eim_forms)
                else:
                    return ParametrizedProblem_DerivedClass.assemble_operator(self, term) # may raise an exception
                    
            @override
            def compute_theta(self, term):
                original_thetas = ParametrizedProblem_DerivedClass.compute_theta(self, term) # may raise an exception
                if term in self.terms:
                    eim_thetas = list()
                    assert len(self.separated_forms[term]) == len(original_thetas)
                    for (form, original_theta) in zip(self.separated_forms[term], original_thetas):
                        # Append coefficients computed with EIM, if applicable
                        for addend in form.coefficients:
                            eim_thetas__list = list()
                            for factor in addend:
                                N_EIM = None
                                if hasattr(self.compute_theta.__func__, "N_EIM"):
                                    assert term in self.compute_theta.__func__.N_EIM and q < len(self.compute_theta.__func__.N_EIM[term])
                                    N_EIM = self.compute_theta.__func__.N_EIM[term][q]
                                eim_thetas__list.append(self.EIM_approximations[factor].compute_interpolated_theta(N_EIM))
                            eim_thetas__cartesian_product = cartesian_product(*eim_thetas__list)
                            for tuple_ in eim_thetas__cartesian_product:
                                eim_thetas_tuple = original_thetas[q]
                                for eim_thata_factor in tuple_:
                                    eim_thetas_tuple *= eim_thata_factor
                                eim_thetas.append(eim_thetas_tuple)
                        # Append coefficients which did not require EIM, if applicable
                        for _ in form._form_unchanged:
                            eim_thetas.append(original_theta)
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
