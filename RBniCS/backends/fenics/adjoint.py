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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl import Form
from dolfin import adjoint as dolfin_adjoint
from RBniCS.utils.decorators import backend_for, tuple_of

@backend_for("FEniCS", inputs=((Form, tuple_of(Form)), ))
def adjoint(arg):
    assert isinstance(arg, (Form, tuple))
    if isinstance(arg, Form):
        return dolfin_adjoint(arg)
    elif isinstance(arg, tuple):
        output = list()
        for a in arg:
            assert isinstance(a, Form)
            output.append(dolfin_adjoint(a))
        return tuple(output)
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to adjoint")
        
