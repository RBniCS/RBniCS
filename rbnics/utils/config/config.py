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
import sys
import configparser
from rbnics.utils.decorators import overload, set_of
from rbnics.utils.mpi import parallel_io

class Config(object):
    rbnics_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
    
    # Set class defaults
    defaults = {
        "backends": {
            "online backend": "numpy",
            "required backends": None
        },
        "EIM": {
            "cache": {"disk", "RAM"},
            "disk cache limit": "unlimited",
            "RAM cache limit": "1"
        },
        "problems": {
            "cache": {"disk", "RAM"},
            "disk cache limit": "unlimited",
            "RAM cache limit": "1"
        },
        "reduced problems": {
            "cache": {"RAM"},
            "RAM cache limit": "unlimited"
        },
        "SCM": {
            "cache": {"disk", "RAM"},
            "disk cache limit": "unlimited",
            "RAM cache limit": "1"
        }
    }
    
    # Read in required backends
    required_backends = set()
    for root, dirs, files in os.walk(os.path.join(rbnics_directory, "backends")):
        for dir_ in dirs:
            if dir_ in sys.modules:
                required_backends.add(dir_)
        break # prevent recursive exploration
    defaults["backends"]["required backends"] = required_backends
    del required_backends
    
    def __init__(self):
        # Setup configparser from defaults
        self._config_as_parser = configparser.ConfigParser()
        self._config_as_parser.optionxform = str
        for (section, options_and_values) in self.defaults.items():
            self._config_as_parser.add_section(section)
            for (option, value) in options_and_values.items():
                self._config_as_parser.set(section, option, self._value_to_parser(section, option, value))
        # Setup dict
        self._config_as_dict = dict()
        self._parser_to_dict()
        
    def read(self, directory=None):
        # Read from configparser
        config_files_list = list()
        config_files_list.append(os.path.join(self.rbnics_directory, ".rbnicsrc"))
        if directory is None:
            if hasattr(sys.modules["__main__"], "__file__") and "pytest" not in sys.modules: # from script
                main_directory = os.path.dirname(os.path.realpath(sys.modules["__main__"].__file__))
            else: # interactive or pytest
                main_directory = os.getcwd()
        else:
            main_directory = directory
        main_directory_split = main_directory.split(os.path.sep)
        for p in range(len(main_directory_split), 0, -1):
            new_config_file_list = list()
            new_config_file_list.append(os.path.sep)
            new_config_file_list.extend(main_directory_split[:p])
            new_config_file_list.append(".rbnicsrc")
            new_config_file = os.path.join(*new_config_file_list)
            if new_config_file not in config_files_list:
                config_files_list.append(new_config_file)
        self._config_as_parser.read(config_files_list)
        # Update dict
        self._parser_to_dict()
        
    def write(self, file_):
        def write_config_parser():
            self._config_as_parser.write(file_)
        parallel_io(write_config_parser)
        
    def get(self, section, option):
        return self._config_as_dict[section][option]
        
    def set(self, section, option, value):
        self._config_as_parser.set(section, option, self._value_to_parser(section, option, value))
        self._config_as_dict[section][option] = value
        
    @overload(str, str, str)
    def _value_to_parser(self, section, option, value):
        assert isinstance(self.defaults[section][option], str)
        return value
        
    @overload(str, str, bool)
    def _value_to_parser(self, section, option, value):
        assert isinstance(self.defaults[section][option], bool)
        return str(value)
        
    @overload(str, str, set_of(str))
    def _value_to_parser(self, section, option, value):
        default = self.defaults[section][option]
        assert isinstance(default, set)
        assert value.issubset(default)
        value_str = ", ".join(str(v) for v in sorted(value))
        if len(value) < 2:
            value_str += "," # to differentiate between str and a set with one element
        return value_str
        
    def _value_from_parser(self, section, option, value):
        assert isinstance(value, str)
        if "," in value:
            assert isinstance(self.defaults[section][option], set)
            value = value.strip(",") # strip trailing comma which has been possibly added to differentiate between str and set
            return set([v.strip() for v in value.split(",")]).difference(("", ))
        else:
            if value.lower() in ("yes", "true", "on"):
                assert isinstance(self.defaults[section][option], bool)
                return True
            elif value.lower() in ("no", "false", "off"):
                assert isinstance(self.defaults[section][option], bool)
                return False
            else:
                assert isinstance(self.defaults[section][option], str)
                return value
            
    def _parser_to_dict(self):
        for section in self._config_as_parser.sections():
            self._config_as_dict[section] = dict()
            for (option, value) in self._config_as_parser.items(section):
                self._config_as_dict[section][option] = self._value_from_parser(section, option, value)
                
    def __eq__(self, other):
        return (
            self._config_as_parser == other._config_as_parser
                and
            self._config_as_dict == other._config_as_dict
        )
        
config = Config()
config.read()
