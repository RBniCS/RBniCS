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
## @file scm.py
#  @brief Implementation of the successive constraints method for the approximation of the coercivity constant
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.utils.decorators import Extends, override, ReducedProblemDecoratorFor
from RBniCS.shape_parametrization.problems.shape_parametrization_decorated_problem import ShapeParametrization

@ReducedProblemDecoratorFor(ShapeParametrization)
def ShapeParametrizationDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):
    #~~~~~~~~~~~~~~~~~~~~~~~~~     SHAPE PARAMETRIZATION CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
    ## @class ShapeParametrizationDecoratedReducedProblem
    #
    # A decorator class that allows to overload methods related to shape parametrization and mesh motion
    @Extends(ParametrizedReducedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class ShapeParametrizationDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
    
        ###########################     CONSTRUCTORS     ########################### 
        ## @defgroup Constructors Methods related to the construction of the SCM object
        #  @{
        
        @override
        def __init__(self, truth_problem, **kwargs):
            # Call the standard initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
                         
        #  @}
        ########################### end - CONSTRUCTORS - end ###########################
    
    # return value (a class) for the decorator
    return ShapeParametrizationDecoratedReducedProblem_Class
    