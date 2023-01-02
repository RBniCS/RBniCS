# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.io import ErrorAnalysisTable, TextBox, TextLine
from rbnics.utils.config import config

def error_analysis_coanda(self, N_max, filename=None, **kwargs):
    
    components = ["u", "p"]  # but not "s"
    kwargs["components"] = components

    print(TextBox(self.truth_problem.name() + " " + self.label + " error analysis begins", fill="="))
    print("")

    error_analysis_table = ErrorAnalysisTable(self.testing_set)
    error_analysis_table.set_Nmax(N_max)
    for component in components:
        error_analysis_table.add_column(
            "error_" + component, group_name="solution_" + component, operations=("mean", "max"))
        error_analysis_table.add_column(
            "relative_error_" + component, group_name="solution_" + component, operations=("mean", "max"))
    error_analysis_table.add_column("error_output", group_name="output", operations=("mean", "max"))
    error_analysis_table.add_column("relative_error_output", group_name="output", operations=("mean", "max"))

    for (n_int, n_arg) in zip(range(1, N_max+1),range(1, N_max+1)):
        print(TextLine(str(n_int), fill="#"))
        self.reduced_problem.flag = False
        self.reduced_problem._solution_cache.clear()
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine(str(mu_index), fill="-"))
            self.reduced_problem.set_mu(mu)
            self.reduced_problem.solve(n_arg, **kwargs)
            error = self.reduced_problem.compute_error(**kwargs)
            relative_error = self.reduced_problem.compute_relative_error(**kwargs)

            self.reduced_problem.compute_output()
            error_output = self.reduced_problem.compute_error_output(**kwargs)
            relative_error_output = self.reduced_problem.compute_relative_error_output(**kwargs)


            for component in components:
                error_analysis_table["error_" + component, n_int, mu_index] = error[component]
                error_analysis_table[
                    "relative_error_" + component, n_int, mu_index] = relative_error[component]

            error_analysis_table["error_output", n_int, mu_index] = error_output
            error_analysis_table["relative_error_output", n_int, mu_index] = relative_error_output

    # Print
    print("")
    print(error_analysis_table)

    print("")
    print(TextBox(self.truth_problem.name() + " " + self.label + " error analysis ends", fill="="))
    print("")

    # Export error analysis table
    error_analysis_table.save(
        self.folder["error_analysis"], "error_analysis" if filename is None else filename)
