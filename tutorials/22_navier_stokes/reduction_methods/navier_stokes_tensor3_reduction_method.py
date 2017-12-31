# Copyright (C) 2015-2018 by the RBniCS authors
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

def NavierStokesTensor3ReductionMethod(NavierStokesReductionMethod_DerivedClass):
    
    NavierStokesTensor3ReductionMethod_Base = NavierStokesReductionMethod_DerivedClass
    
    class NavierStokesTensor3ReductionMethod_Class(NavierStokesTensor3ReductionMethod_Base):
        pass
        
    # return value (a class) for the decorator
    return NavierStokesTensor3ReductionMethod_Class
