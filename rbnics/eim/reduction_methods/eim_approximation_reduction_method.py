# Copyright (C) 2015-2019 by the RBniCS authors
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

import os
from rbnics.reduction_methods.base import ReductionMethod
from rbnics.backends import abs, evaluate, max
from rbnics.utils.decorators import snapshot_links_to_cache
from rbnics.utils.io import ErrorAnalysisTable, Folders, GreedySelectedParametersList, GreedyErrorEstimatorsList, SpeedupAnalysisTable, TextBox, TextLine, Timer
from rbnics.utils.test import PatchInstanceMethod

# Empirical interpolation method for the interpolation of parametrized functions
class EIMApproximationReductionMethod(ReductionMethod):
    
    # Default initialization of members
    def __init__(self, EIM_approximation):
        # Call the parent initialization
        ReductionMethod.__init__(self, EIM_approximation.folder_prefix)
        
        # $$ OFFLINE DATA STRUCTURES $$ #
        # High fidelity problem
        self.EIM_approximation = EIM_approximation
        # Declare a new container to store the snapshots
        self.snapshots_container = self.EIM_approximation.parametrized_expression.create_snapshots_container()
        self._training_set_parameters_to_snapshots_container_index = dict()
        # I/O
        self.folder["snapshots"] = os.path.join(self.folder_prefix, "snapshots")
        self.folder["post_processing"] = os.path.join(self.folder_prefix, "post_processing")
        self.greedy_selected_parameters = GreedySelectedParametersList()
        self.greedy_errors = GreedyErrorEstimatorsList()
        #
        # By default set a tolerance slightly larger than zero, in order to
        # stop greedy iterations in trivial cases by default
        self.tol = 1e-15
    
    def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
        import_successful = ReductionMethod.initialize_training_set(self, self.EIM_approximation.mu_range, ntrain, enable_import, sampling, **kwargs)
        # Since exact evaluation is required, we cannot use a distributed training set
        self.training_set.serialize_maximum_computations()
        # Also initialize the map from parameter values to snapshots container index
        self._training_set_parameters_to_snapshots_container_index = dict((mu, mu_index) for (mu_index, mu) in enumerate(self.training_set))
        return import_successful
        
    def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
        return ReductionMethod.initialize_testing_set(self, self.EIM_approximation.mu_range, ntest, enable_import, sampling, **kwargs)
    
    # Perform the offline phase of EIM
    def offline(self):
        need_to_do_offline_stage = self._init_offline()
        if need_to_do_offline_stage:
            self._offline()
        self._finalize_offline()
        return self.EIM_approximation
        
    # Initialize data structures required for the offline phase
    def _init_offline(self):
        # Prepare folders and init EIM approximation
        required_folders = Folders()
        required_folders.update(self.folder)
        required_folders.update(self.EIM_approximation.folder)
        optional_folders = Folders()
        optional_folders["cache"] = required_folders.pop("cache") # this does not affect the availability of offline data
        optional_folders["testing_set"] = required_folders.pop("testing_set") # this is required only in the error/speedup analysis
        optional_folders["error_analysis"] = required_folders.pop("error_analysis") # this is required only in the error analysis
        optional_folders["speedup_analysis"] = required_folders.pop("speedup_analysis") # this is required only in the speedup analysis
        at_least_one_required_folder_created = required_folders.create()
        at_least_one_optional_folder_created = optional_folders.create()  # noqa: F841
        if not at_least_one_required_folder_created:
            return False # offline construction should be skipped, since data are already available
        else:
            self.EIM_approximation.init("offline")
            return True # offline construction should be carried out
        
    @snapshot_links_to_cache
    def _offline(self):
        interpolation_method_name = self.EIM_approximation.parametrized_expression.interpolation_method_name()
        description = self.EIM_approximation.parametrized_expression.description()
        
        # Evaluate the parametrized expression for all parameters in the training set
        print(TextBox(interpolation_method_name + " preprocessing phase begins for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        for (mu_index, mu) in enumerate(self.training_set):
            print(TextLine(interpolation_method_name + " " + str(mu_index), fill=":"))
            
            self.EIM_approximation.set_mu(mu)
            
            print("evaluate parametrized expression at mu =", mu)
            self.EIM_approximation.evaluate_parametrized_expression()
            self.EIM_approximation.export_solution(self.folder["snapshots"], "truth_" + str(mu_index))
            
            print("add to snapshots")
            self.add_to_snapshots(self.EIM_approximation.snapshot)

            print("")
            
        # If basis generation is POD, compute the first POD modes of the snapshots
        if self.EIM_approximation.basis_generation == "POD":
            print("compute basis")
            N_POD = self.compute_basis_POD()
            print("")
        
        print(TextBox(interpolation_method_name + " preprocessing phase ends for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        print(TextBox(interpolation_method_name + " offline phase begins for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        if self.EIM_approximation.basis_generation == "Greedy":
            # Initialize first parameter to be used
            (error_max, relative_error_max) = self.greedy()
            print("initial maximum interpolation error =", error_max)
            print("initial maximum interpolation relative error =", relative_error_max)
            
            print("")
            
            # Carry out greedy selection
            while self.EIM_approximation.N < self.Nmax and relative_error_max >= self.tol:
                print(TextLine(interpolation_method_name + " N = " + str(self.EIM_approximation.N), fill=":"))
                
                self._print_greedy_interpolation_solve_message()
                self.EIM_approximation.solve()
                
                print("compute and locate maximum interpolation error")
                self.EIM_approximation.snapshot = self.load_snapshot()
                (error, maximum_error, maximum_location) = self.EIM_approximation.compute_maximum_interpolation_error()
                
                print("update locations with", maximum_location)
                self.update_interpolation_locations(maximum_location)
                
                print("update basis")
                self.update_basis_greedy(error, maximum_error)
                
                print("update interpolation matrix")
                self.update_interpolation_matrix()
                
                (error_max, relative_error_max) = self.greedy()
                print("maximum interpolation error =", error_max)
                print("maximum interpolation relative error =", relative_error_max)
                
                print("")
        else:
            while self.EIM_approximation.N < N_POD:
                print(TextLine(interpolation_method_name + " N = " + str(self.EIM_approximation.N), fill=":"))
            
                print("solve interpolation for basis number", self.EIM_approximation.N)
                self.EIM_approximation._solve(self.EIM_approximation.basis_functions[self.EIM_approximation.N])
                
                print("compute and locate maximum interpolation error")
                self.EIM_approximation.snapshot = self.EIM_approximation.basis_functions[self.EIM_approximation.N]
                (error, maximum_error, maximum_location) = self.EIM_approximation.compute_maximum_interpolation_error()
                
                print("update locations with", maximum_location)
                self.update_interpolation_locations(maximum_location)
                
                self.EIM_approximation.N += 1
                
                print("update interpolation matrix")
                self.update_interpolation_matrix()
                
                print("")
                
        print(TextBox(interpolation_method_name + " offline phase ends for" + "\n" + "\n".join(description), fill="="))
        print("")
        
    # Finalize data structures required after the offline phase
    def _finalize_offline(self):
        self.EIM_approximation.init("online")
        
    def _print_greedy_interpolation_solve_message(self):
        print("solve interpolation for mu =", self.EIM_approximation.mu)
        
    # Update the snapshots container
    def add_to_snapshots(self, snapshot):
        self.snapshots_container.enrich(snapshot)
        
    # Update basis (greedy version)
    def update_basis_greedy(self, error, maximum_error):
        if abs(maximum_error) > 0.:
            self.EIM_approximation.basis_functions.enrich(error/maximum_error)
        else:
            # Trivial case, greedy will stop at the first iteration
            assert self.EIM_approximation.N == 0
            self.EIM_approximation.basis_functions.enrich(error) # error is actually zero
        self.EIM_approximation.basis_functions.save(self.EIM_approximation.folder["basis"], "basis")
        self.EIM_approximation.N += 1

    # Update basis (POD version)
    def compute_basis_POD(self):
        POD = self.EIM_approximation.parametrized_expression.create_POD_container()
        POD.store_snapshot(self.snapshots_container)
        (_, _, basis_functions, N) = POD.apply(self.Nmax, self.tol)
        self.EIM_approximation.basis_functions.enrich(basis_functions)
        self.EIM_approximation.basis_functions.save(self.EIM_approximation.folder["basis"], "basis")
        # do not increment self.EIM_approximation.N
        POD.print_eigenvalues(N)
        POD.save_eigenvalues_file(self.folder["post_processing"], "eigs")
        POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy")
        return N
        
    def update_interpolation_locations(self, maximum_location):
        self.EIM_approximation.interpolation_locations.append(maximum_location)
        self.EIM_approximation.interpolation_locations.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_locations")
    
    # Assemble the interpolation matrix
    def update_interpolation_matrix(self):
        self.EIM_approximation.interpolation_matrix[0] = evaluate(self.EIM_approximation.basis_functions[:self.EIM_approximation.N], self.EIM_approximation.interpolation_locations)
        self.EIM_approximation.interpolation_matrix.save(self.EIM_approximation.folder["reduced_operators"], "interpolation_matrix")
            
    # Load the precomputed snapshot
    def load_snapshot(self):
        assert self.EIM_approximation.basis_generation == "Greedy"
        mu = self.EIM_approximation.mu
        mu_index = self._training_set_parameters_to_snapshots_container_index[mu]
        assert mu == self.training_set[mu_index]
        return self.snapshots_container[mu_index]
        
    # Choose the next parameter in the offline stage in a greedy fashion
    def greedy(self):
        assert self.EIM_approximation.basis_generation == "Greedy"
        
        # Print some additional information on the consistency of the reduced basis
        if self.EIM_approximation.N > 0: # skip during initialization
            self.EIM_approximation.solve()
            self.EIM_approximation.snapshot = self.load_snapshot()
            error = self.EIM_approximation.snapshot - self.EIM_approximation.basis_functions*self.EIM_approximation._interpolation_coefficients
            error_on_interpolation_locations = evaluate(error, self.EIM_approximation.interpolation_locations)
            (maximum_error, _) = max(abs(error))
            (maximum_error_on_interpolation_locations, _) = max(abs(error_on_interpolation_locations)) # for consistency check, should be zero
            print("interpolation error for current mu =", abs(maximum_error))
            print("interpolation error on interpolation locations for current mu =", abs(maximum_error_on_interpolation_locations))
        
        # Carry out the actual greedy search
        def solve_and_computer_error(mu):
            self.EIM_approximation.set_mu(mu)
            
            self.EIM_approximation.solve()
            self.EIM_approximation.snapshot = self.load_snapshot()
            (_, maximum_error, _) = self.EIM_approximation.compute_maximum_interpolation_error()
            return abs(maximum_error)
            
        if self.EIM_approximation.N == 0:
            print("find initial mu")
        else:
            print("find next mu")
        (error_max, error_argmax) = self.training_set.max(solve_and_computer_error)
        self.EIM_approximation.set_mu(self.training_set[error_argmax])
        self.greedy_selected_parameters.append(self.training_set[error_argmax])
        self.greedy_selected_parameters.save(self.folder["post_processing"], "mu_greedy")
        self.greedy_errors.append(error_max)
        self.greedy_errors.save(self.folder["post_processing"], "error_max")
        if abs(self.greedy_errors[0]) > 0.:
            return (abs(error_max), abs(error_max/self.greedy_errors[0]))
        else:
            # Trivial case, greedy should stop after one iteration after having store a zero basis function
            assert len(self.greedy_errors) in (1, 2)
            if len(self.greedy_errors) == 1:
                assert self.EIM_approximation.N == 0
                # Tweak the tolerance to force getting in the greedy loop
                self.tol = -1.
            elif len(self.greedy_errors) == 2:
                assert error_max == 0.
                assert self.EIM_approximation.N == 1
                # Tweak back the tolerance to force getting out of the greedy loop
                assert self.tol == -1.
                self.tol = 1.
            return (0., 0.)
    
    # Compute the error of the empirical interpolation approximation with respect to the
    # exact function over the testing set
    def error_analysis(self, N_generator=None, filename=None, **kwargs):
        assert len(kwargs) == 0 # not used in this method
            
        self._init_error_analysis(**kwargs)
        self._error_analysis(N_generator, filename, **kwargs)
        self._finalize_error_analysis(**kwargs)
        
    def _error_analysis(self, N_generator=None, filename=None, **kwargs):
        if N_generator is None:
            def N_generator(n):
                return n
                
        N = self.EIM_approximation.N
        interpolation_method_name = self.EIM_approximation.parametrized_expression.interpolation_method_name()
        description = self.EIM_approximation.parametrized_expression.description()
        
        print(TextBox(interpolation_method_name + " error analysis begins for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        error_analysis_table = ErrorAnalysisTable(self.testing_set)
        error_analysis_table.set_Nmax(N)
        error_analysis_table.add_column("error", group_name="eim", operations=("mean", "max"))
        error_analysis_table.add_column("relative_error", group_name="eim", operations=("mean", "max"))
        
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine(interpolation_method_name + " " + str(mu_index), fill=":"))
            
            self.EIM_approximation.set_mu(mu)
            
            # Evaluate the exact function on the truth grid
            self.EIM_approximation.evaluate_parametrized_expression()
            
            for n in range(1, N + 1): # n = 1, ... N
                n_arg = N_generator(n)
                
                if n_arg is not None:
                    self.EIM_approximation.solve(n_arg)
                    (_, error, _) = self.EIM_approximation.compute_maximum_interpolation_error(n)
                    (_, relative_error, _) = self.EIM_approximation.compute_maximum_interpolation_relative_error(n)
                    error_analysis_table["error", n, mu_index] = abs(error)
                    error_analysis_table["relative_error", n, mu_index] = abs(relative_error)
                else:
                    error_analysis_table["error", n, mu_index] = NotImplemented
                    error_analysis_table["relative_error", n, mu_index] = NotImplemented
        
        # Print
        print("")
        print(error_analysis_table)
        
        print("")
        print(TextBox(interpolation_method_name + " error analysis ends for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        # Export error analysis table
        error_analysis_table.save(self.folder["error_analysis"], "error_analysis" if filename is None else filename)
        
    # Compute the speedup of the empirical interpolation approximation with respect to the
    # exact function over the testing set
    def speedup_analysis(self, N_generator=None, filename=None, **kwargs):
        assert len(kwargs) == 0 # not used in this method
            
        self._init_speedup_analysis(**kwargs)
        self._speedup_analysis(N_generator, filename, **kwargs)
        self._finalize_speedup_analysis(**kwargs)
        
    # Initialize data structures required for the speedup analysis phase
    def _init_speedup_analysis(self, **kwargs):
        # Make sure to clean up snapshot cache to ensure that parametrized
        # expression evaluation is actually carried out
        self.EIM_approximation._snapshot_cache.clear()
        # ... and also disable the capability of importing/exporting truth solutions
        def disable_import_solution_method(self_, folder=None, filename=None, solution=None):
            raise OSError
        self.disable_import_solution = PatchInstanceMethod(self.EIM_approximation, "import_solution", disable_import_solution_method)
        def disable_export_solution_method(self_, folder=None, filename=None, solution=None):
            pass
        self.disable_export_solution = PatchInstanceMethod(self.EIM_approximation, "export_solution", disable_export_solution_method)
        self.disable_import_solution.patch()
        self.disable_export_solution.patch()
        
    def _speedup_analysis(self, N_generator=None, filename=None, **kwargs):
        if N_generator is None:
            def N_generator(n):
                return n
                
        N = self.EIM_approximation.N
        interpolation_method_name = self.EIM_approximation.parametrized_expression.interpolation_method_name()
        description = self.EIM_approximation.parametrized_expression.description()
        
        print(TextBox(interpolation_method_name + " speedup analysis begins for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        speedup_analysis_table = SpeedupAnalysisTable(self.testing_set)
        speedup_analysis_table.set_Nmax(N)
        speedup_analysis_table.add_column("speedup", group_name="speedup", operations=("min", "mean", "max"))
        
        evaluate_timer = Timer("parallel")
        EIM_timer = Timer("serial")
        
        for (mu_index, mu) in enumerate(self.testing_set):
            print(TextLine(interpolation_method_name + " " + str(mu_index), fill=":"))
            
            self.EIM_approximation.set_mu(mu)
            
            # Evaluate the exact function on the truth grid
            evaluate_timer.start()
            self.EIM_approximation.evaluate_parametrized_expression()
            elapsed_evaluate = evaluate_timer.stop()
            
            for n in range(1, N + 1): # n = 1, ... N
                n_arg = N_generator(n)
                
                if n_arg is not None:
                    EIM_timer.start()
                    self.EIM_approximation.solve(n_arg)
                    elapsed_EIM = EIM_timer.stop()
                    speedup_analysis_table["speedup", n, mu_index] = elapsed_evaluate/elapsed_EIM
                else:
                    speedup_analysis_table["speedup", n, mu_index] = NotImplemented
        
        # Print
        print("")
        print(speedup_analysis_table)
        
        print("")
        print(TextBox(interpolation_method_name + " speedup analysis ends for" + "\n" + "\n".join(description), fill="="))
        print("")
        
        # Export speedup analysis table
        speedup_analysis_table.save(self.folder["speedup_analysis"], "speedup_analysis" if filename is None else filename)
        
    # Finalize data structures required after the speedup analysis phase
    def _finalize_speedup_analysis(self, **kwargs):
        # Restore the capability to import/export truth solutions
        self.disable_import_solution.unpatch()
        self.disable_export_solution.unpatch()
        del self.disable_import_solution
        del self.disable_export_solution
