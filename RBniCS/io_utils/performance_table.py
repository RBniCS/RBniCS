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
## @file performance_table.py
#  @brief Type to store performance of the reduced order method.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy import zeros as Content
from numpy import min, exp, mean, log, max

class PerformanceTable(object):
    def __init__(self, xi_test):
        self._columns = dict() # string to Content matrix
        self._columns_operations = dict() # string to tuple
        self._groups = dict() # string to list
        self._group_names_sorted = list()
        self._len_xi_test = len(xi_test)
        self._Nmin = 1
        self._Nmax = 0
        
    def set_Nmin(self, Nmin):
        self._Nmin = Nmin
        
    def set_Nmax(self, Nmax):
        self._Nmax = Nmax
            
    def add_column(self, column_name, group_name, operations):
        assert self._Nmax > 0
        assert self._Nmax >= self._Nmin
        self._columns[column_name] = Content((self._Nmax - self._Nmin + 1, self._len_xi_test))
        if group_name not in self._groups:
            self._groups[group_name] = list()
            self._group_names_sorted.append(group_name) # preserve the ordering provided by the user
        self._groups[group_name].append(column_name)
        if isinstance(operations, str):
            self._columns_operations[column_name] = (operations,)
        elif isinstance(operations, tuple):
            self._columns_operations[column_name] = operations
        else:
            raise ValueError("Invalid operation in PerformanceTable")
                
    def __getitem__(self, args):
        assert len(args) == 3
        return self._columns[args[0]][args[1] - self._Nmin, args[2]]
        
    def __setitem__(self, args, value):
        assert len(args) == 3
        self._columns[args[0]][args[1] - self._Nmin, args[2]] = value
            
    def __str__(self):
        output = ""
        for group in self._group_names_sorted:
            table_index = list() # of strings
            table_header = dict() # from string to string
            table_content = dict() # from string to Content array
            column_size = dict() # from string to int
            # First column should be the reduced space dimension
            table_index.append("N")
            table_header["N"] = "N"
            table_content["N"] = range(self._Nmin, self._Nmax + 1)
            column_size["N"] = max([max([len(str(x)) for x in table_content["N"]]), len("N")])
            # Then fill in with postprocessed data
            for column in self._groups[group]:
                for operation in self._columns_operations[column]:
                    # Set header
                    if operation == "min" or operation == "max":
                        current_table_header = operation + "(" + column + ")"
                        current_table_index = operation + "_" + column
                    elif operation == "mean":
                        current_table_header = "gmean(" + column + ")"
                        current_table_index = "gmean_" + column
                    else:
                        raise ValueError("Invalid operation in PerformanceTable")
                    table_index.append(current_table_index)
                    table_header[current_table_index] = current_table_header
                    # Compute the required operation of each column over the second index (test set)
                    table_content[current_table_index] = Content((self._Nmax - self._Nmin + 1,))
                    for n in range(self._Nmin, self._Nmax + 1):
                        if operation == "min":
                            current_table_content = min(self._columns[column][n - self._Nmin, :])
                        elif operation == "mean":
                            current_table_content = exp(mean(log(self._columns[column][n - self._Nmin, :])))
                        elif operation == "max":
                            current_table_content = max(self._columns[column][n - self._Nmin, :])
                        else:
                            raise ValueError("Invalid operation in PerformanceTable")
                        table_content[current_table_index][n - self._Nmin] = current_table_content
                    # Get the width of the columns
                    column_size[current_table_index] = max([max([len(str(x)) for x in table_content[current_table_index]]), len(current_table_header)])
            # Prepare formetter for string conversion
            formatter = ""
            for i in range(len(table_index)):
                formatter += "{" + str(i) + ":<{" + table_index[i] + "}}"
                if i < len(table_index) - 1:
                    formatter += "\t"
            # Print the header
            current_line = list()
            for t in table_index:
                current_line.append(table_header[t])
            output += formatter.format(*current_line, **column_size) + "\n"
            # Print the content
            for n in range(self._Nmin, self._Nmax + 1):
                current_line = list()
                for t in table_index:
                    current_line.append(table_content[t][n - self._Nmin])
                output += formatter.format(*current_line, **column_size) + "\n"
            output += "\n"
        return output[:-2] # remove the last two newlines
        
