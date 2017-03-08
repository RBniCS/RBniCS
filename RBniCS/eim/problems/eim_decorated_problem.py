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
## @file eim.py
#  @brief Implementation of the empirical interpolation method for the interpolation of parametrized functions
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from itertools import product as cartesian_product
from RBniCS.backends import ParametrizedExpressionFactory, SeparatedParametrizedForm
from RBniCS.utils.decorators import Extends, override, ProblemDecoratorFor
from RBniCS.eim.utils.io import AffineExpansionSeparatedFormsStorage
from RBniCS.eim.problems.eim_approximation import EIMApproximation

def EIMDecoratedProblem(
    basis_generation="Greedy",
    **decorator_kwargs
):
    from RBniCS.eim.problems.exact_parametrized_functions import ExactParametrizedFunctions
    from RBniCS.eim.problems.eim import EIM
    
    @ProblemDecoratorFor(EIM, ExactAlgorithm=ExactParametrizedFunctions)
    def EIMDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):
                
        @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
        class EIMDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            
            ## Default initialization of members
            @override
            def __init__(self, V, **kwargs):
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Storage for EIM reduced problems
                self.separated_forms = dict() # from terms to AffineExpansionSeparatedFormsStorage
                self.EIM_approximations = dict() # from coefficients to EIMApproximation
                
                # Store value of N_EIM passed to solve
                self._N_EIM = None
                            
                # Avoid useless assignments
                self._update_N_EIM__previous_kwargs = None
                
            @override
            def set_mu_range(self, mu_range):
                # Call to parent
                ParametrizedDifferentialProblem_DerivedClass.set_mu_range(self, mu_range)
                
                # Preprocess each term in the affine expansions
                # Note that this cannot be done in __init__ because operators may depend on self.mu,
                # which is not defined at __init__ time
                for term in self.terms:
                    forms = ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term)
                    Q = len(forms)
                    self.separated_forms[term] = AffineExpansionSeparatedFormsStorage(Q)
                    for q in range(Q):
                        self.separated_forms[term][q] = SeparatedParametrizedForm(forms[q])
                        self.separated_forms[term][q].separate()
                        # All parametrized coefficients should be approximated by EIM
                        for (addend_index, addend) in enumerate(self.separated_forms[term][q].coefficients):
                            for (factor, factor_name) in zip(addend, self.separated_forms[term][q].placeholders_names(addend_index)):
                                if factor not in self.EIM_approximations:
                                    self.EIM_approximations[factor] = EIMApproximation(self, ParametrizedExpressionFactory(self, factor), type(self).__name__ + "/eim/" + factor_name, basis_generation)
                
            @override
            def solve(self, **kwargs):
                self._update_N_EIM(**kwargs)
                return ParametrizedDifferentialProblem_DerivedClass.solve(self, **kwargs)
            
            def _update_N_EIM(self, **kwargs):
                if kwargs != self._update_N_EIM__previous_kwargs:
                    if "EIM" in kwargs:
                        self._N_EIM = dict()
                        N_EIM = kwargs["EIM"]
                        for term in self.separated_forms:
                            self._N_EIM[term] = list()
                            if isinstance(N_EIM, dict):
                                assert term in N_EIM
                                assert len(N_EIM[term]) == len(self.separated_forms[term])
                                for N_eim_term_q in N_EIM[term]:
                                    self._N_EIM[term].append(N_eim_term_q)
                            else:
                                assert isinstance(N_EIM, int)
                                for _ in self.separated_forms[term]:
                                    self._N_EIM[term].append(N_EIM)
                    else:
                        self._N_EIM = None
                    self._update_N_EIM__previous_kwargs = kwargs
                
                
            ###########################     PROBLEM SPECIFIC     ########################### 
            ## @defgroup ProblemSpecific Problem specific methods
            #  @{
            
            @override
            def assemble_operator(self, term):
                if term in self.terms:
                    eim_forms = list()
                    for form in self.separated_forms[term]:
                        # Append forms computed with EIM, if applicable
                        for (index, addend) in enumerate(form.coefficients):
                            replacements__list = list()
                            for factor in addend:
                                replacements__list.append(self.EIM_approximations[factor].Z)
                            replacements__cartesian_product = cartesian_product(*replacements__list)
                            for new_coeffs in replacements__cartesian_product:
                                eim_forms.append(
                                    form.replace_placeholders(index, new_coeffs)
                                )
                        # Append forms which did not require EIM, if applicable
                        for unchanged_form in form.unchanged_forms:
                            eim_forms.append(unchanged_form)
                    return tuple(eim_forms)
                else:
                    return ParametrizedDifferentialProblem_DerivedClass.assemble_operator(self, term) # may raise an exception
                    
            @override
            def compute_theta(self, term):
                original_thetas = ParametrizedDifferentialProblem_DerivedClass.compute_theta(self, term) # may raise an exception
                if term in self.terms:
                    eim_thetas = list()
                    assert len(self.separated_forms[term]) == len(original_thetas)
                    if self._N_EIM is not None:
                        assert term in self._N_EIM 
                        assert len(self.separated_forms[term]) == len(self._N_EIM[term])
                    for (q, (form, original_theta)) in enumerate(zip(self.separated_forms[term], original_thetas)):
                        # Append coefficients computed with EIM, if applicable
                        for addend in form.coefficients:
                            eim_thetas__list = list()
                            for factor in addend:
                                N_EIM = None
                                if self._N_EIM is not None:
                                    N_EIM = self._N_EIM[term][q]
                                eim_thetas__list.append(self.EIM_approximations[factor].compute_interpolated_theta(N_EIM))
                            eim_thetas__cartesian_product = cartesian_product(*eim_thetas__list)
                            for tuple_ in eim_thetas__cartesian_product:
                                eim_thetas_tuple = original_thetas[q]
                                for eim_thata_factor in tuple_:
                                    eim_thetas_tuple *= eim_thata_factor
                                eim_thetas.append(eim_thetas_tuple)
                        # Append coefficients which did not require EIM, if applicable
                        for _ in form.unchanged_forms:
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
