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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

# Auxiliary function: provide a common interface to __getitem__ for FunctionsList and BasisFunctionsMatrix
def functions_list_basis_functions_matrix_adapter(functions, backend):
    assert isinstance(functions, (backend.FunctionsList, backend.BasisFunctionsMatrix))
    if isinstance(functions, backend.FunctionsList):
        output = [function for function in functions]
        return (output, len(output))
    elif isinstance(functions, backend.BasisFunctionsMatrix):
        output = list()
        for (basis_component_index, component_name) in sorted(functions._basis_component_index_to_component_name.iteritems()):
            for function in functions._components[component_name]:
                output.append(function)
        return (output, functions._component_name_to_basis_component_length)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in functions_list_basis_functions_matrix_adapter.")
        
