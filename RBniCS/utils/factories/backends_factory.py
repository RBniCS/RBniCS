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
## @file reduced_problem_factory.py
#  @brief Factory to generate a reduced problem corresponding to a given reduction method and truth problem
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

#~~~~~~~~~~~~~~~~~~~~~~~~~     PARAMETRIZED PROBLEM BASE CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
## @class ReducedProblemFactory
#

from RBniCS.utils.decorators import ReducedProblemFor, ReducedProblemDecoratorFor
from RBniCS.utils.io import log, DEBUG

# Factory to combine all available backends
def backends_factory(backends_module):
    # Logging
    log(DEBUG,
        "In backends_factory with\n" +
        "\tbackends module = " + str(backends_module)
    )
    
    # Function to add attribute to backends module
    def add_backends_module_attr(input_map, return_map):
        for r in return_map["Abstract"]:
            def backend_selector(*args, **kwargs):
                input_types = list()
                input_types.extend([type(arg) for arg in args])
                input_types.extend([type(kwargs[key]) for key in kwargs])
                for i in input_map:
                    if i is input_types:
                        if not input_map[i] in backends_factory._disabled_backends:
                            return return_map[ input_map[i] ][r](*args, **kwargs)
                return TypeError("No backend found for return type " + r + " with input arguments " + input_types)
            setattr(backends_module, r, backend_selector)
    
    # Generate backend classes
    add_backends_module_attr(BackendFor._all_classes_inputs, BackendFor._all_classes)
    
    # Generate backend functions
    add_backends_module_attr(backend_for._all_functions_inputs, backend_for._all_functions)
    
    # Logging
    log(DEBUG,
        "After backends_factory, backends module contains\n" +
        "\t" + dir(backends_module)
    )
    
backends_factory._disabled_backends = list() # of strings

def enable_backend(library):
    pass # They are enabled by default when the decorator process them
    
def disable_backend(library):
    assert library not in backends_factory._disabled_backends
    backends_factory._disabled_backends.append(library)
    
# Generate Online* classes
def set_online_backend(library, backends_module):
    # Logging
    log(DEBUG,
        "In set_online_backend with\n" +
        "\tlibrary = " + library +
        "\tbackends module = " + str(backends_module)
    )
    
    if not hasattr(backends_module, "_online_backend"):
        setattr(backends_module, "_online_backend", library) # do not hardcode library in the function, it may change afterwards!
        #
        online_types = dict()
        online_types["Matrix"] = "OnlineMatrix"
        online_types["Vector"] = "OnlineVector"
        online_types["Function"] = "OnlineFunction"
        online_types["AffineExpansionStorage"] = "AffineExpansionOnlineStorage"
        # TODO serve anche eigen solver per POD
        # TODO ripensaci, non capisco bene la differenza tra tipi e le funzioni che li generano
        return_map = BackendFor._all_classes
        for t in online_types:
            def backend_selector(*args, **kwargs):
                if not hasattr(backend_selector, "_online_backend") or backend_selector._online_backend is not online_backend:
                    backend_selector._online_backend = library
                    backend_selector._type = # TODO
                return return_map[backends_module._online_backend][t](*args, **kwargs)
            setattr(backends_module, online_types[t], backend_selector)
    else: # just replace the string, the other attributes have been already set
        setattr(backends_module, "_online_backend", library)
    
    # Logging
    log(DEBUG,
        "After set_online_backend, backends module contains\n" +
        "\t" + dir(backends_module)
    )
    
