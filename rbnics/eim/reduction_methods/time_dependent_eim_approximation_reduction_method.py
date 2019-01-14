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


from rbnics.sampling import ParameterSpaceSubset
from rbnics.sampling.distributions import DiscreteDistribution, UniformDistribution
from rbnics.eim.problems.time_dependent_eim_approximation import EnlargedMu
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod

class TimeDependentEIMApproximationReductionMethod(EIMApproximationReductionMethod):
    
    def __init__(self, EIM_approximation):
        # Call the parent initialization
        EIMApproximationReductionMethod.__init__(self, EIM_approximation)
        
    def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
        import_successful = EIMApproximationReductionMethod.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
        # Initialize time training set
        time_training_set = ParameterSpaceSubset()
        # Test if can import
        time_import_successful = False
        if enable_import:
            time_import_successful = time_training_set.load(self.folder["training_set"], "time_training_set") and (len(time_training_set) == ntrain)
        if not time_import_successful:
            time_sampling = self._generate_time_sampling(**kwargs)
            time_training_set.generate([(0., self.EIM_approximation.T)], ntrain, time_sampling)
            # Export
            time_training_set.save(self.folder["training_set"], "time_training_set")
        # Combine both sets into one
        self._combine_sets(self.training_set, time_training_set)
        # Also initialize the map from parameter values to snapshots container index
        self._training_set_parameters_to_snapshots_container_index = {(mu["mu"], mu["t"]): mu_index for (mu_index, mu) in enumerate(self.training_set)}
        # Return
        assert time_import_successful == import_successful
        return import_successful
        
    def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
        import_successful = EIMApproximationReductionMethod.initialize_testing_set(self, ntest, enable_import, sampling, **kwargs)
        # Initialize time testing set
        time_testing_set = ParameterSpaceSubset()
        # Test if can import
        time_import_successful = False
        if enable_import:
            time_import_successful = time_testing_set.load(self.folder["testing_set"], "time_testing_set") and (len(time_testing_set) == ntest)
        if not import_successful:
            time_sampling = self._generate_time_sampling(**kwargs)
            try:
                t0 = self.EIM_approximation.truth_problem._time_stepping_parameters["monitor"]["initial_time"]
            except KeyError:
                t0 = self.t0
            T = self.EIM_approximation.T
            time_testing_set.generate([(t0, T)], ntest, time_sampling)
            # Export
            time_testing_set.save(self.folder["testing_set"], "time_testing_set")
        # Combine both sets into one
        self._combine_sets(self.testing_set, time_testing_set)
        # Return
        assert time_import_successful == import_successful
        return import_successful
        
    def _generate_time_sampling(self, **kwargs):
        if "time_sampling" in kwargs:
            time_sampling = kwargs["time_sampling"]
        else:
            time_sampling = UniformDistribution()
        try:
            dt = self.EIM_approximation.truth_problem._time_stepping_parameters["monitor"]["time_step_size"]
        except KeyError:
            assert self.EIM_approximation.dt is not None
            dt = self.EIM_approximation.dt
        return DiscreteDistribution(time_sampling, (dt, ))
        
    def _combine_sets(self, mu_set, time_set):
        for (n, (mu, t)) in enumerate(zip(mu_set, time_set)):
            mu_t = EnlargedMu()
            mu_t["mu"] = mu
            assert len(t) == 1
            mu_t["t"] = t[0]
            mu_set[n] = mu_t
            
    def _print_greedy_interpolation_solve_message(self):
        print("solve interpolation for mu =", self.EIM_approximation.mu, "and t =", self.EIM_approximation.t)
        
    # Load the precomputed snapshot. Overridden to correct the assert
    def load_snapshot(self):
        assert self.EIM_approximation.basis_generation == "Greedy"
        mu = self.EIM_approximation.mu
        t = self.EIM_approximation.t
        mu_index = self._training_set_parameters_to_snapshots_container_index[(mu, t)]
        assert mu == self.training_set[mu_index]["mu"]
        assert t == self.training_set[mu_index]["t"]
        return self.snapshots_container[mu_index]
