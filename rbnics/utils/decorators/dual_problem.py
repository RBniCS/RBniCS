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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.utils.decorators.extends import Extends
from rbnics.utils.decorators.override import override
from rbnics.utils.decorators.sync_setters import sync_setters

def DualProblem(ParametrizedDifferentialProblem_DerivedClass):
            
    @Extends(ParametrizedDifferentialProblem_DerivedClass, preserve_class_name=True)
    class DualProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        
        ## Default initialization of members.
        @override
        @sync_setters("primal_problem", "set_mu", "mu")
        @sync_setters("primal_problem", "set_mu_range", "mu_range")
        def __init__(self, primal_problem):
            # Call to parent
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, primal_problem)
            
            # Store the primal problem
            self.primal_problem = primal_problem
            
            # Change the folder names in Parent
            new_folder_prefix = primal_problem.folder_prefix + "/" + "dual"
            for (key, name) in self.folder.iteritems():
                self.folder[key] = name.replace(self.folder_prefix, new_folder_prefix)
            self.folder_prefix = new_folder_prefix
            
    # return value (a class) for the decorator
    return DualProblem_Class
    
