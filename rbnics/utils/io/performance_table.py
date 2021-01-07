# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import sys
import collections
from numpy import exp, isnan, log, max, mean, min, nan, zeros as Content
from rbnics.utils.io.csv_io import CSVIO
from rbnics.utils.io.folders import Folders


class PerformanceTable(object):

    # Storage for class methods
    _suppressed_groups = list()
    _preprocessor_setitem = dict()

    def __init__(self, testing_set):
        self._columns = dict()  # string to Content matrix
        self._columns_operations = dict()  # string to tuple
        self._columns_not_implemented = dict()  # string to bool
        self._rows_not_implemented = dict()  # string to dict of bool
        self._groups = dict()  # string to list
        self._group_names_sorted = list()
        self._len_testing_set = len(testing_set)
        self._Nmin = 1
        self._Nmax = 0

    def set_Nmin(self, Nmin):
        self._Nmin = Nmin

    def set_Nmax(self, Nmax):
        self._Nmax = Nmax

    def add_column(self, column_name, group_name, operations):
        assert self._Nmax > 0
        assert self._Nmax >= self._Nmin
        assert column_name not in self._columns and column_name not in self._columns_operations
        self._columns[column_name] = Content((self._Nmax - self._Nmin + 1, self._len_testing_set))
        self._columns_not_implemented[column_name] = None  # will be set to a bool
        self._rows_not_implemented[column_name] = {
            n: None for n in range(self._Nmax - self._Nmin + 1)}  # will be set to a bool
        if group_name not in self._groups:
            self._groups[group_name] = list()
            self._group_names_sorted.append(group_name)  # preserve the ordering provided by the user
        self._groups[group_name].append(column_name)
        if isinstance(operations, str):
            self._columns_operations[column_name] = (operations,)
        elif isinstance(operations, tuple):
            self._columns_operations[column_name] = operations
        else:
            raise ValueError("Invalid operation in PerformanceTable")

    @classmethod
    def suppress_group(cls, group_name):
        cls._suppressed_groups.append(group_name)

    @classmethod
    def clear_suppressed_groups(cls):
        cls._suppressed_groups = list()

    @classmethod
    def preprocess_setitem(cls, group_name, function):
        cls._preprocessor_setitem[group_name] = function

    @classmethod
    def clear_setitem_preprocessing(cls):
        cls._preprocessor_setitem.clear()

    def __getitem__(self, args):
        assert len(args) == 3
        column_name = args[0]
        N = args[1]
        mu_index = args[2]
        assert self._columns_not_implemented[column_name] in (True, False)
        assert self._rows_not_implemented[column_name][N - self._Nmin] in (True, False)
        if (not self._columns_not_implemented[column_name]
                and not self._rows_not_implemented[column_name][N - self._Nmin]):
            return self._columns[column_name][N - self._Nmin, mu_index]
        else:
            return CustomNotImplementedAfterDiv

    def __setitem__(self, args, value):
        assert len(args) == 3
        column_name = args[0]
        N = args[1]
        mu_index = args[2]
        if is_not_implemented(value):
            assert self._columns_not_implemented[column_name] in (None, True, False)
            if self._columns_not_implemented[column_name] is None:
                self._columns_not_implemented[column_name] = True
            assert self._rows_not_implemented[column_name][N - self._Nmin] in (None, True)
            if self._rows_not_implemented[column_name][N - self._Nmin] is None:
                self._rows_not_implemented[column_name][N - self._Nmin] = True
        else:
            assert self._columns_not_implemented[column_name] in (None, True, False)
            if self._columns_not_implemented[column_name] in (None, True):
                self._columns_not_implemented[column_name] = False
            assert self._rows_not_implemented[column_name][N - self._Nmin] in (None, False)
            if self._rows_not_implemented[column_name][N - self._Nmin] is None:
                self._rows_not_implemented[column_name][N - self._Nmin] = False
            if column_name not in self._preprocessor_setitem:
                self._columns[column_name][N - self._Nmin, mu_index] = value
            else:
                self._columns[column_name][N - self._Nmin, mu_index] = self._preprocessor_setitem[column_name](value)

    def _process(self):
        groups_content = collections.OrderedDict()
        for group in self._group_names_sorted:
            # Skip suppresed groups
            if group in self._suppressed_groups:
                continue
            # Populate all columns
            columns = list()
            for column in self._groups[group]:
                assert self._columns_not_implemented[column] in (True, False)
                if self._columns_not_implemented[column] is False:
                    columns.append(column)
            if len(columns) == 0:
                continue
            # Storage for print
            table_index = list()  # of strings
            table_header = dict()  # from string to string
            table_content = dict()  # from string to Content array
            column_size = dict()  # from string to int
            # First column should be the reduced space dimension
            table_index.append("N")
            table_header["N"] = "N"
            table_content["N"] = list(range(self._Nmin, self._Nmax + 1))
            column_size["N"] = max([max([len(str(x)) for x in table_content["N"]]), len("N")])
            # Then fill in with postprocessed data
            for column in columns:
                for operation in self._columns_operations[column]:
                    # Set header
                    if operation in ("min", "max"):
                        current_table_header = operation + "(" + column + ")"
                        current_table_index = operation + "_" + column
                    elif operation == "mean":
                        current_table_header = "gmean(" + column + ")"
                        current_table_index = "gmean_" + column
                    else:
                        raise ValueError("Invalid operation in PerformanceTable")
                    table_index.append(current_table_index)
                    table_header[current_table_index] = current_table_header
                    # Compute the required operation of each column over the second index (testing set)
                    table_content[current_table_index] = Content((self._Nmax - self._Nmin + 1,))
                    for n in range(self._Nmin, self._Nmax + 1):
                        assert self._rows_not_implemented[column][n - self._Nmin] in (None, True, False)
                        if self._rows_not_implemented[column][n - self._Nmin] is False:
                            if operation == "min":
                                current_table_content = min(self._columns[column][n - self._Nmin, :])
                            elif operation == "mean":
                                data = self._columns[column][n - self._Nmin, :]
                                if not data.any():  # all zeros
                                    current_table_content = 0.
                                else:
                                    data[data == 0.] = sys.float_info.epsilon
                                    current_table_content = exp(mean(log(data)))
                            elif operation == "max":
                                current_table_content = max(self._columns[column][n - self._Nmin, :])
                            else:
                                raise ValueError("Invalid operation in PerformanceTable")
                            table_content[current_table_index][n - self._Nmin] = current_table_content
                        else:
                            table_content[current_table_index][n - self._Nmin] = nan
                    # Get the width of the columns
                    column_size[current_table_index] = max([max([
                        len(str(x)) for x in table_content[current_table_index]]), len(current_table_header)])
            # Save content
            assert group not in groups_content
            groups_content[group] = (table_index, table_header, table_content, column_size)
        return groups_content

    def __str__(self):
        groups_content = self._process()
        output = ""
        for (group, (table_index, table_header, table_content, column_size)) in groups_content.items():
            table_index_without_N = table_index[1:]
            # Prepare formatter for string conversion
            formatter = ""
            for (column_index, column_name) in enumerate(table_index):
                formatter += "{" + str(column_index) + ":<{" + column_name + "}}"
                if column_index < len(table_index) - 1:
                    formatter += "\t"
            # Print the header
            current_line = list()
            for t in table_index:
                current_line.append(table_header[t])
            output += formatter.format(*current_line, **column_size) + "\n"
            # Print the current row, only if its content was set to NotImplemented
            for n in range(self._Nmin, self._Nmax + 1):
                current_line = list()
                all_not_implemented = all(isnan(
                    table_content[t][n - self._Nmin]) for t in table_index_without_N)
                assert any(isnan(
                    table_content[t][n - self._Nmin]) for t in table_index_without_N) is all_not_implemented
                if not all_not_implemented:
                    for t in table_index:
                        current_line.append(table_content[t][n - self._Nmin])
                    output += formatter.format(*current_line, **column_size) + "\n"
            output += "\n"
        return output[:-2]  # remove the last two newlines

    def save(self, directory, filename):
        full_directory = Folders.Folder(os.path.join(str(directory), filename))
        full_directory.create()
        groups_content = self._process()
        for (group, (table_index, table_header, table_content, _)) in groups_content.items():
            table_index_without_N = table_index[1:]
            current_file = list()
            # Store the header
            current_file.append([table_header[t] for t in table_index])
            # Store the current row, only if its content was set to NotImplemented
            for n in range(self._Nmin, self._Nmax + 1):
                all_not_implemented = all(isnan(
                    table_content[t][n - self._Nmin]) for t in table_index_without_N)
                assert any(isnan(
                    table_content[t][n - self._Nmin]) for t in table_index_without_N) is all_not_implemented
                if not all_not_implemented:
                    current_file.append([table_content[t][n - self._Nmin] for t in table_index])
            # Save
            CSVIO.save_file(current_file, full_directory, group)

    def load(self, directory, filename):
        raise RuntimeError("PerformanceTable.load has not been implemented yet")


class CustomNotImplementedType(object):
    def __init__(self):
        pass


CustomNotImplemented = CustomNotImplementedType()


def is_not_implemented(value):
    if value is NotImplemented:
        return True
    elif value is CustomNotImplemented:
        return True
    elif hasattr(value, "__iter__"):
        each_is_not_implemented = [is_not_implemented(v) for v in value]
        assert all(b == each_is_not_implemented[0] for b in each_is_not_implemented)
        return each_is_not_implemented[0]
    else:
        return False


class CustomNotImplementedAfterDivType(CustomNotImplementedType):
    def __init__(self):
        pass

    def __truediv__(self, other):
        return CustomNotImplemented

    def __rtruediv__(self, other):
        return CustomNotImplemented

    def __itruediv__(self, other):
        return CustomNotImplemented


CustomNotImplementedAfterDiv = CustomNotImplementedAfterDivType()
