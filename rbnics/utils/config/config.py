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

import os
import sys
import configparser

rbnics_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

def init_config(config):
    """
    Set default options
    """
    
    config.add_section("backends")
    config.set("backends", "delay assembly", "False")
    config.set("backends", "online backend", "numpy")
    config.set("backends", "required backends", []) # will be filled in automatically

    config.add_section("EIM")
    config.set("EIM", "cache", ["Disk", "RAM"])
    
    config.add_section("problems")
    config.set("problems", "cache", ["Disk", "RAM"])
    
    config.add_section("reduced problems")
    config.set("reduced problems", "cache", ["RAM"])
    
    config.add_section("SCM")
    config.set("SCM", "cache", ["Disk", "RAM"])

def read_config(config):
    """
    Possibly read customized options from file
    """
    
    # Read from file
    config_files_list = list()
    config_files_list.append(os.path.join(rbnics_directory, ".rbnicsrc"))
    if hasattr(sys.modules["__main__"], "__file__"): # from script
        main_directory = os.path.dirname(os.path.realpath(sys.modules["__main__"].__file__))
    else: # interactive
        main_directory = os.getcwd()
    main_directory_split = main_directory.split(os.path.sep)
    for p in range(len(main_directory_split), 0, -1):
        new_config_file_list = list()
        new_config_file_list.append(os.path.sep)
        new_config_file_list.extend(main_directory_split[:p])
        new_config_file_list.append(".rbnicsrc")
        new_config_file = os.path.join(*new_config_file_list)
        if new_config_file not in config_files_list:
            config_files_list.append(new_config_file)
    config.read(config_files_list)
    
    # Convert list of string options
    _convert_list_str_options(config, "EIM", "cache")
    assert set(config.get("EIM", "cache")).issubset(["Disk", "RAM"])
    _convert_list_str_options(config, "problems", "cache")
    assert set(config.get("problems", "cache")).issubset(["Disk", "RAM"])
    _convert_list_str_options(config, "reduced problems", "cache")
    assert set(config.get("reduced problems", "cache")).issubset(["RAM"])
    _convert_list_str_options(config, "SCM", "cache")
    assert set(config.get("SCM", "cache")).issubset(["Disk", "RAM"])
    
    # Fill in ("backends", "required backends")
    required_backends = set()
    for root, dirs, files in os.walk(os.path.join(rbnics_directory, "backends")):
        for dir_ in dirs:
            if dir_ in sys.modules:
                required_backends.add(dir_)
        break # prevent recursive exploration
    config.set("backends", "required backends", required_backends)
    
def _convert_list_str_options(config, section, option):
    value = config.get(section, option)
    assert isinstance(value, (list, str))
    if isinstance(value, str):
        config.set(section, option, list(map(str.strip, value.replace("\n", ",").split(","))))
    assert isinstance(config.get(section, option), list)
        
config = configparser.ConfigParser()
init_config(config)
read_config(config)

del rbnics_directory
