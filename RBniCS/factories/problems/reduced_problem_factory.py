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
## @file reduced_problem_factory.py
#  @brief Factory to associate reduced problem classes to reduction method classes.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedProblemFactory
#

# Factory to associate reduced problem classes to reduction method classes.
def ReducedProblemFactory(truth_problem, reduction_method):
    from RBniCS.reduction_methods import EllipticCoercivePODGalerkinReduction, EllipticCoerciveRBReduction
    from RBniCS.problems import EllipticCoercivePODGalerkinReducedProblem, EllipticCoerciveRBReducedProblem
        
    # Determine whether RB or POD-Galerkin method is used
    def _ReducedProblem_TypeFactory(reduction_method):
        if isinstance(reduction_method, EllipticCoercivePODGalerkinReduction):
            return EllipticCoercivePODGalerkinReducedProblem
        elif isinstance(reduction_method, EllipticCoerciveRBReduction):
            return EllipticCoerciveRBReducedProblem
        else:
            raise TypeError("Invalid arguments in ReducedProblemFactory.")
            
    # Combine them
    @_DecoratedReducedProblem_TypeFactory(truth_problem)
    class _ReducedProblem_Type( _ReducedProblem_TypeFactory(reduction_method) ):
        pass
        
    # Create an instance of the generated class
    reduced_problem = _ReducedProblem_Type(truth_problem)
    # Return
    return reduced_problem
    
# Decorator to add EIM or SCM, as required
def _DecoratedReducedProblem_TypeFactory(truth_problem):
    from RBniCS.eim.problems import EIMDecoratedReducedProblem
    from RBniCS.eim.problems import ExactParametrizedFunctionEvaluationDecoratedReducedProblem
    from RBniCS.scm.problems import SCMDecoratedReducedProblem
    from RBniCS.scm.problems import ExactCoercivityConstantDecoratedReducedProblem
    from RBniCS.shape_parametrization.problems import ShapeParametrizationDecoratedReducedProblem
    
    def _DecoratedReducedProblem_TypeFactory__Decorator(ParametrizedProblem_DerivedClass):
        DecoratedReducedProblem_Type = ParametrizedProblem_DerivedClass
        if hasattr(truth_problem, "_problem_decorators"):
            problem_decorators = truth_problem._problem_decorators
            if "EIM" in problem_decorators and problem_decorators["EIM"]:
                DecoratedReducedProblem_Type = EIMDecoratedReducedProblem(DecoratedReducedProblem_Type)
            if "ExactParametrizedFunctionEvaluation" in problem_decorators and problem_decorators["ExactParametrizedFunctionEvaluation"]:
                DecoratedReducedProblem_Type = ExactParametrizedFunctionEvaluationDecoratedReducedProblem(DecoratedReducedProblem_Type)
            if "SCM" in problem_decorators and problem_decorators["SCM"]:
                DecoratedReducedProblem_Type = SCMDecoratedReducedProblem(DecoratedReducedProblem_Type)
            if "ExactCoercivityConstant" in problem_decorators and problem_decorators["ExactCoercivityConstant"]:
                DecoratedReducedProblem_Type = ExactCoercivityConstantDecoratedReducedProblem(DecoratedReducedProblem_Type)
            if "ShapeParametrization" in problem_decorators and problem_decorators["ShapeParametrization"]:
                DecoratedReducedProblem_Type = ShapeParametrizationDecoratedReducedProblem(DecoratedReducedProblem_Type)
        return DecoratedReducedProblem_Type
    return _DecoratedReducedProblem_TypeFactory__Decorator
