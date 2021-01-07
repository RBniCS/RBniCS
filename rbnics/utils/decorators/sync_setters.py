# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.test import PatchInstanceMethod


def sync_setters__internal(other_object__name__or__instance, method__name, private_attribute__name,
                           method__decorator=None):

    def sync_setters_decorator(__init__):

        def __synced__init__(self, *args, **kwargs):
            # Call the parent initialization
            __init__(self, *args, **kwargs)
            # Get other_object
            if isinstance(other_object__name__or__instance, str):
                other_object = getattr(self, other_object__name__or__instance)
            else:
                other_object = other_object__name__or__instance
            # Sync setters only if the other object is not None
            if other_object is not None:
                # Initialize private storage
                if method__name not in _synced_setters:
                    _synced_setters[method__name] = dict()
                if method__name not in _original_setters:
                    _original_setters[method__name] = dict()
                # Detect if either self or other_object are already in sync with somebody else
                all_synced_setters_for_method_self = None
                if self in _synced_setters[method__name]:
                    all_synced_setters_for_method_self = _synced_setters[method__name][self]
                all_synced_setters_for_method_other_object = None
                if other_object in _synced_setters[method__name]:
                    all_synced_setters_for_method_other_object = _synced_setters[method__name][other_object]
                # Add current methods to the set of syncronized setters in storage
                if (all_synced_setters_for_method_self is not None
                        and all_synced_setters_for_method_other_object is not None):
                    assert all_synced_setters_for_method_self is all_synced_setters_for_method_other_object
                elif (all_synced_setters_for_method_self is None
                        and all_synced_setters_for_method_other_object is not None):
                    _original_setters[method__name][self] = getattr(self, method__name)
                    all_synced_setters_for_method = all_synced_setters_for_method_other_object
                    all_synced_setters_for_method.add(self)
                    _synced_setters[method__name][self] = all_synced_setters_for_method
                elif (all_synced_setters_for_method_self is not None
                        and all_synced_setters_for_method_other_object is None):
                    _original_setters[method__name][other_object] = getattr(other_object, method__name)
                    all_synced_setters_for_method = all_synced_setters_for_method_self
                    all_synced_setters_for_method.add(other_object)
                    _synced_setters[method__name][other_object] = all_synced_setters_for_method
                else:
                    _original_setters[method__name][self] = getattr(self, method__name)
                    _original_setters[method__name][other_object] = getattr(other_object, method__name)
                    all_synced_setters_for_method = set()
                    all_synced_setters_for_method.add(self)
                    all_synced_setters_for_method.add(other_object)
                    _synced_setters[method__name][self] = all_synced_setters_for_method
                    _synced_setters[method__name][other_object] = all_synced_setters_for_method
                # Now both storage and local variable should be consistent between self and other_object,
                # and pointing to the same memory location
                assert _synced_setters[method__name][self] is _synced_setters[method__name][other_object]

                # Override both self and other_object setters to propagate to all synced setters
                def overridden_method(self_, arg):
                    if method__name not in _synced_setters__disabled_methods:
                        all_synced_setters = _synced_setters[method__name][self_]
                        for obj in all_synced_setters:
                            setter = _original_setters[method__name][obj]
                            if getattr(obj, private_attribute__name) is not arg:
                                setter(arg)

                method__decorator__changed = False
                if method__decorator is not None:
                    if method__name not in _synced_setters__decorators:
                        _synced_setters__decorators[method__name] = list()
                    if isinstance(method__decorator, list):
                        for method__decorator_i in method__decorator:
                            if method__decorator_i not in _synced_setters__decorators[method__name]:
                                _synced_setters__decorators[method__name].append(method__decorator_i)
                                method__decorator__changed = True
                    else:
                        if method__decorator not in _synced_setters__decorators[method__name]:
                            _synced_setters__decorators[method__name].append(method__decorator)
                        method__decorator__changed = True
                    for method__decorator_i in _synced_setters__decorators[method__name]:
                        overridden_method = method__decorator_i(overridden_method)
                if all_synced_setters_for_method_self is None or method__decorator__changed:
                    PatchInstanceMethod(self, method__name, overridden_method).patch()
                if all_synced_setters_for_method_other_object is None or method__decorator__changed:
                    PatchInstanceMethod(other_object, method__name, overridden_method).patch()
                # Make sure that the value of my attribute is in sync with the value that is currently
                # stored in other_object, because it was set before overriding was carried out
                getattr(self, method__name)(getattr(other_object, private_attribute__name))

        return __synced__init__

    return sync_setters_decorator


def sync_setters(other_object, method__name, private_attribute__name, method__decorator=None):
    assert method__name in (
        "set_final_time", "set_initial_time", "set_mu", "set_mu_range", "set_time", "set_time_step_size")
    if method__name in ("set_final_time", "set_initial_time", "set_mu", "set_time", "set_time_step_size"):
        return sync_setters__internal(other_object, method__name, private_attribute__name, method__decorator)
    elif method__name == "set_mu_range":
        if method__decorator is not None:
            method__decorator = [method__decorator, set_mu_range__decorator]
        else:
            method__decorator = set_mu_range__decorator
        return sync_setters__internal(other_object, method__name, private_attribute__name, method__decorator)
    else:
        raise ValueError("Invalid method in sync_setters.")


def set_mu_range__decorator(set_mu_range__method):

    def set_mu_range__decorated(self_, mu_range):
        # set_mu_range by defaults calls set_mu. Since set_mu
        # (1) requires a properly initialized mu range, but
        # (2) it has been overridden to be kept in sync, also
        #     for object which have not been initialized yet,
        # we first disable set_mu
        _synced_setters__disabled_methods.add("set_mu")
        # We set (and sync) the mu range
        set_mu_range__method(self_, mu_range)
        # Finally, we restore the original set_mu and set (and sync)
        # the value of mu so that it has the correct length,
        # as done in ParametrizedProblem
        _synced_setters__disabled_methods.remove("set_mu")
        self_.set_mu(tuple([r[0] for r in mu_range]))

    return set_mu_range__decorated


_original_setters = dict()
_synced_setters = dict()
_synced_setters__decorators = dict()
_synced_setters__disabled_methods = set()
