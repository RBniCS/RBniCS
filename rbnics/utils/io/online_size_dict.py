# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import OrderedDict
from rbnics.utils.decorators import dict_of, overload


class OnlineSizeDict(OrderedDict):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super(OnlineSizeDict, self).__init__(*args, **kwargs)

    @staticmethod
    def generate_from_N_and_kwargs(components_, default, N, **kwargs):
        # need to add underscore to components_ becuase "components" is also a possible kwargs key
        if len(components_) > 1:
            if N is None:
                all_components_in_kwargs = components_[0] in kwargs
                for component in components_:
                    if all_components_in_kwargs:
                        assert component in kwargs, (
                            "You need to specify the online size of all components in kwargs")
                    else:
                        assert component not in kwargs, (
                            "You need to specify the online size of all components in kwargs")
                if all_components_in_kwargs:
                    N = OnlineSizeDict()
                    for component in components_:
                        N[component] = kwargs[component]
                        del kwargs[component]
                else:
                    assert isinstance(default, dict)
                    N = OnlineSizeDict(default)  # copy the default dict
            else:
                assert isinstance(N, (int, OnlineSizeDict))
                if isinstance(N, int):
                    N_int = N
                    N = OnlineSizeDict()
                    for component in components_:
                        N[component] = N_int
                        assert component not in kwargs, "You cannot provide both an int and kwargs for components"
                elif isinstance(N, OnlineSizeDict):
                    # check that components are the same, and are ordered correctly
                    assert list(N.keys()) == list(default.keys())
                else:
                    raise TypeError("Invalid N")
        else:
            assert len(components_) == 1
            component_0 = components_[0]
            if N is None:
                if component_0 in kwargs:
                    N_int = kwargs[component_0]
                else:
                    assert isinstance(default, int)
                    N_int = default
            else:
                assert isinstance(N, int)
                N_int = N
            N = OnlineSizeDict()
            N[component_0] = N_int

        return N, kwargs

    def __getitem__(self, k):
        return super(OnlineSizeDict, self).__getitem__(k)

    def __setitem__(self, k, v):
        return super(OnlineSizeDict, self).__setitem__(k, v)

    def __delitem__(self, k):
        return super(OnlineSizeDict, self).__delitem__(k)

    def get(self, k, default=None):
        return super(OnlineSizeDict, self).get(k, default)

    def setdefault(self, k, default=None):
        return super(OnlineSizeDict, self).setdefault(k, default)

    def pop(self, k):
        return super(OnlineSizeDict, self).pop(k)

    def update(self, **kwargs):
        super(OnlineSizeDict, self).update(**kwargs)

    def __contains__(self, k):
        return super(OnlineSizeDict, self).__contains__(k)

    # Override N += N_bc so that it is possible to increment online size due to boundary conditions
    # (several components)
    @overload(lambda cls: cls)
    def __iadd__(self, other):
        for key in self:
            self[key] += other[key]
        return self

    # Override N += N_bc so that it is possible to increment online size due to boundary conditions
    # (single component)
    @overload(int)
    def __iadd__(self, other):
        assert len(self) == 1
        for key in self:
            self[key] += other
        return self

    # Override N + N_bc as well
    def __add__(self, other):
        output = OnlineSizeDict(self)
        output += other
        return output

    # Override __eq__ so that it is possible to check equality of dictionary with an int
    @overload(int)
    def __eq__(self, other):
        for (key, value) in self.items():
            if value != other:
                return False
        return True

    @overload((lambda cls: cls, dict_of(str, int)))
    def __eq__(self, other):
        return super(OnlineSizeDict, self).__eq__(other)

    # Override __ne__ so that it is possible to check not equality of dictionary with an int
    @overload(int)
    def __ne__(self, other):
        for (key, value) in self.items():
            if value == other:
                return False
        return True

    @overload((lambda cls: cls, dict_of(str, int)))
    def __ne__(self, other):
        return super(OnlineSizeDict, self).__ne__(other)

    # Override __lt__ so that it is possible to check if dictionary is less than an int
    @overload(int)
    def __lt__(self, other):
        for (key, value) in self.items():
            if value >= other:
                return False
        return True

    @overload((lambda cls: cls, dict_of(str, int)))
    def __lt__(self, other):
        return super(OnlineSizeDict, self).__lt__(other)

    # Override __gt__ so that it is possible to check if dictionary is greater than an int
    @overload(int)
    def __gt__(self, other):
        for (key, value) in self.items():
            if value <= other:
                return False
        return True

    @overload((lambda cls: cls, dict_of(str, int)))
    def __gt__(self, other):
        return super(OnlineSizeDict, self).__gt__(other)

    # Override __str__ to print an integer if all values are the same
    def __str__(self):
        if len(set(self.values())) == 1:
            for (_, value) in self.items():
                break
            return str(value)
        else:
            return "{" + ", ".join([key + ": " + str(value) for (key, value) in self.items()]) + "}"
