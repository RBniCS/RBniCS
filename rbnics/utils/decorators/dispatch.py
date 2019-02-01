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

import inspect
import itertools
from collections import OrderedDict
from numpy import ndarray as array
import multipledispatch.conflict
from multipledispatch.core import dispatch as original_dispatch, ismethod
from multipledispatch.dispatcher import Dispatcher as OriginalDispatcher

# == Signature to string == #
def str_signature(sig):
    str_sig = ", ".join(c.__name__ if isinstance(c, type) else str(c) for c in sig)
    if not bool(str_sig): # empty string
        return "None"
    else:
        return str_sig
        
# == Exception for unavailable signature == #
class UnavailableSignatureError(NotImplementedError):
    def __init__(self, name, available_signatures, unavailable_signature):
        error_message = "Could not find signature for " + name + ".\n"
        error_message += "Available signatures are:\n"
        for available_signature in available_signatures:
            error_message += "\t" + str_signature(available_signature) + "\n"
        error_message += "Requested unavailable signature is:\n"
        error_message += "\t" + str_signature(unavailable_signature) + "\n"
        NotImplementedError.__init__(self, error_message)

# == Exception for ambiguous signature == #
class AmbiguousSignatureError(NotImplementedError):
    def __init__(self, name, available_signatures, ambiguous_signatures):
        error_message = "Ambiguous signature signature for " + name + ".\n"
        error_message += "Available signatures are:\n"
        for available_signature in available_signatures:
            error_message += "\t" + str_signature(available_signature) + "\n"
        error_message += "Ambiguous signatures are:\n"
        for ambiguous_signature_pair in ambiguous_signatures:
            error_message += "\t" + str_signature(ambiguous_signature_pair[0]) + " and " + str_signature(ambiguous_signature_pair[1]) + "\n"
        NotImplementedError.__init__(self, error_message)
        
def ambiguity_error(dispatcher, ambiguities):
    """
    Raise error when ambiguity is detected
    Parameters
    ----------
    dispatcher : Dispatcher
        The dispatcher on which the ambiguity was detected
    ambiguities : set
        Set of type signature pairs that are ambiguous within this dispatcher
    """
    raise AmbiguousSignatureError(dispatcher.name, dispatcher.funcs.keys(), ambiguities)
    
# == Exception for invalid signature == #
class InvalidSignatureError(TypeError):
    def __init__(self, name, invalid_signature):
        str_sig = str_signature(invalid_signature)
        error_message = "Invalid dispatch types for " + name + ".\n"
        error_message += "Invalid signature is:\n"
        error_message += "\t" + str_sig + "\n"
        TypeError.__init__(self, error_message)

# == Customize Dispatcher == #
class Dispatcher(OriginalDispatcher):
    __slots__ = '__name__', 'name', 'funcs', '_ordering', '_cache', 'doc', 'signature_to_provided_signature' # extended with new private members
    
    def __init__(self, name, doc=None):
        OriginalDispatcher.__init__(self, name, doc)
        self.signature_to_provided_signature = dict()
        
    def add(self, signature, func, replaces=None, replaces_if=None):
        for types in expand_tuples(signature):
            self._add(types, signature, func, replaces, replaces_if)
        # Trigger reordering, if needed
        self._cache.clear()
        try:
            del self._ordering
        except AttributeError:
            pass
        
    def _add(self, types, provided_signature, func, replaces=None, replaces_if=None):
        try:
            validate_types(types, allow_lambda=False)
        except AssertionError:
            raise InvalidSignatureError(self.name, types)
        
        if replaces is None:
            if types in self.funcs:
                len_types = len(types)
                assert len_types == len(provided_signature)
                previously_provided_signature = self.signature_to_provided_signature[types]
                assert len_types == len(previously_provided_signature)
                old_one_matches = sum(supercedes((previously_provided_signature[i], ), (types[i], )) for i in range(len_types))
                new_one_matches = sum(supercedes((provided_signature[i], ), (types[i], )) for i in range(len_types))
                if old_one_matches > new_one_matches:
                    pass
                elif new_one_matches > old_one_matches:
                    self.funcs[types] = func
                    self.signature_to_provided_signature[types] = provided_signature
                else:
                    raise AmbiguousSignatureError(self.name, self.funcs.keys(), {(types, types)})
            else:
                self.funcs[types] = func
                self.signature_to_provided_signature[types] = provided_signature
        else:
            assert types in self.funcs
            assert self.funcs[types] is replaces
            if replaces_if is None:
                self.funcs[types] = func
                self.signature_to_provided_signature[types] = provided_signature
            else:
                def conditional_func(*args, **kwargs):
                    if replaces_if(*args, **kwargs):
                        return func(*args, **kwargs)
                    else:
                        return replaces(*args, **kwargs)
                self.funcs[types] = conditional_func
                self.signature_to_provided_signature[types] = provided_signature
    add.__doc__ = OriginalDispatcher.add.__doc__ + \
        """
        
        This is a customization of the Dispatcher.add method provided by the multipledispatch
        package so that:
            * replacement of existing signatures is not done silently as in the original library.
              It is only possible to replace a previously added function/method with the same signature
              by means of replaces and replaces_if input arguments.
        """
        
    def reorder(self, on_ambiguity=ambiguity_error):
        return OriginalDispatcher.reorder(self, on_ambiguity)
    
    def __call__(self, *args, **kwargs):
        func = self._get_func(*args)
        return func(*args, **kwargs)
        
    def _get_func(self, *args):
        if len(args) > 1:
            types = get_types(args)
        elif len(args) == 1 and args[0] is not None:
            types = (get_type(args[0]), )
        else:
            types = tuple()
        try:
            func = self._cache[types]
        except KeyError:
            func = self.dispatch(*types)
            if func is None:
                raise UnavailableSignatureError(self.name, self.funcs.keys(), types)
            self._cache[types] = func
        return func
    _get_func.__doc__ = \
        """
        This is a customization required by Dispatcher.__call__ method so that:
            * get_types() function is used to get input types. This handles the case of
              array_of, dict_of, iterable_of, list_of, set_of, tuple_of
            * a custom UnavailableSignatureError is thrown if no corresponding signature is provided
        It is based on the original multipledispatch implementation of Dispatcher.__call__
        """
        
    def dispatch_iter(self, *types):
        for signature in self.ordering:
            if supercedes(types, signature):
                result = self.funcs[signature]
                yield result
    dispatch_iter.__doc__ = \
        """
        This is a customization of the Dispatcher.dispatch_iter method provided by the multipledispatch
        package so that the custom implementation of supercedes() is used
        """
        
    @property
    def __doc__(self):
        return OriginalDispatcher.__doc__.fget(self) # properties are not inherited

        
# == Customize MethodDispatcher == #
class MethodDispatcher_Wrapper(object):
    __slots__ = 'name', 'standard_funcs', 'lambda_funcs', 'dispatchers'
    
    def __init__(self, name, doc=None):
        self.name = name
        self.standard_funcs = OrderedDict()
        self.lambda_funcs = OrderedDict()
        self.dispatchers = OrderedDict()
        
    def add(self, signature, func):
        for types in expand_tuples(signature):
            self._add(types, signature, func)
        
    def _add(self, types, provided_signature, func):
        if any(islambda(typ) for typ in types):
            try:
                validate_types(types, allow_lambda=True)
            except AssertionError:
                raise InvalidSignatureError(self.name, types)
            
            if types in self.lambda_funcs:
                raise AmbiguousSignatureError(self.name, self.lambda_funcs.keys(), {(types, types)})
                
            self.lambda_funcs[types] = func
        else: # no lambda functions
            try:
                validate_types(types, allow_lambda=False)
            except AssertionError:
                raise InvalidSignatureError(self.name, types)
                
            # Delay proper ambiguity checking to MethodDispatcher, other than a simple check
            # on overwriting existing storage
            if (types, provided_signature) in self.standard_funcs:
                raise AmbiguousSignatureError(self.name, self.standard_funcs.keys(), {(types, types)})
                
            self.standard_funcs[(types, provided_signature)] = func
            
    def register(self, *types, **kwargs):
        def _(func):
            self.add(types, func, **kwargs)
            return func
        return _
        
    def __get__(self, instance, owner):
        # Create a new dispatcher based on the current class. This is required to not share dispatcher among inherited classes
        assert owner is not None
        try:
            dispatcher = self.dispatchers[owner]
        except KeyError:
            dispatcher = MethodDispatcher(self, owner, self.name)
            self.dispatchers[owner] = dispatcher
        # Assign self.obj (it may be None if method is called as Class.method(instance, ...) rather than instance.method(...))
        dispatcher.obj = instance
        # Return
        return dispatcher
        
class MethodDispatcher(Dispatcher):
    __slots__ = '__name__', 'name', 'funcs', '_ordering', '_cache', 'doc', 'signature_to_provided_signature', 'origin', 'obj' # extended with new private members
    
    def __init__(self, origin, cls, name, doc=None):
        Dispatcher.__init__(self, name, doc)
        self.origin = origin
        self.obj = None
        # Copy standard and lambda functions
        standard_funcs = origin.standard_funcs.copy()
        lambda_funcs = origin.lambda_funcs.copy()
        # Add all functions from ancestor classes
        for base_cls in inspect.getmro(cls):
            if base_cls is cls:
                continue
            if hasattr(base_cls, name):
                parent_func = getattr(base_cls, name)
                if isinstance(parent_func, MethodDispatcher):
                    parent_func = parent_func.origin
                    for (parent_standard_key_i, parent_standard_func_i) in parent_func.standard_funcs.items():
                        add_to_standard_funcs = True
                        for standard_key_j in standard_funcs.keys():
                            if parent_standard_key_i == standard_key_j:
                                add_to_standard_funcs = False
                        for lambda_key_j in lambda_funcs.keys():
                            signature_lambda_key_j = tuple([typ(cls) if islambda(typ) else typ for typ in lambda_key_j])
                            if parent_standard_key_i[0] == signature_lambda_key_j:
                                add_to_standard_funcs = False
                        if add_to_standard_funcs:
                            standard_funcs[parent_standard_key_i] = parent_standard_func_i
                    for (parent_lambda_key_i, parent_lambda_func_i) in parent_func.lambda_funcs.items():
                        add_to_lambda_funcs = True
                        signature_parent_lambda_key_i = tuple([typ(cls) if islambda(typ) else typ for typ in parent_lambda_key_i])
                        for standard_key_j in standard_funcs.keys():
                            if signature_parent_lambda_key_i == standard_key_j[0]:
                                add_to_lambda_funcs = False
                        for lambda_key_j in lambda_funcs.keys():
                            signature_lambda_key_j = tuple([typ(cls) if islambda(typ) else typ for typ in lambda_key_j])
                            if signature_parent_lambda_key_i == signature_lambda_key_j:
                                add_to_lambda_funcs = False
                        if add_to_lambda_funcs:
                            lambda_funcs[parent_lambda_key_i] = parent_lambda_func_i
                elif ismethod(parent_func) and not hasattr(parent_func, "__isabstractmethod__"):
                    add_to_standard_funcs = True
                    parent_signature = (len(inspect.signature(parent_func).parameters) - 1)*(object, )
                    parent_standard_key = (parent_signature, parent_signature)
                    for standard_key in standard_funcs.keys():
                        if parent_standard_key == standard_funcs:
                            add_to_standard_funcs = False
                    for lambda_key in lambda_funcs:
                        signature_lambda_key = tuple([typ(cls) if islambda(typ) else typ for typ in lambda_key])
                        if parent_signature == signature_lambda_key:
                            add_to_standard_funcs = False
                    if add_to_standard_funcs:
                        standard_funcs[parent_standard_key] = parent_func
                else:
                    pass # This happens with slot wrapper, method wrapper and method descriptor types for builtin objects
        # Add all standard functions
        for (key, func) in standard_funcs.items():
            self._add(key[0], key[1], func)
        # Add all lambda functions
        # NOTE: it is not possible to change the underlying type returned by lambda after this loop has been processed
        for (signature_lambda, lambda_func) in lambda_funcs.items():
            signature = tuple([typ(cls) if islambda(typ) else typ for typ in signature_lambda])
            self._add(signature, signature, lambda_func)
        # Trigger reordering, if needed
        self._cache.clear()
        try:
            del self._ordering
        except AttributeError:
            pass
        
    def __call__(self, *args, **kwargs):
        if self.obj is not None: # called as instance.method(...)
            obj = self.obj
        else: # called as Class.method(instance, ...)
            obj = args[0]
            args = args[1:]
        func = self._get_func(*args)
        return func(obj, *args, **kwargs)
        
    @property
    def __doc__(self):
        return Dispatcher.__doc__.fget(self) # properties are not inherited

# == Customize @dispatch == #
def dispatch(*types, **kwargs):
    name = kwargs.get("name", None)
    module_kwarg = kwargs.get("module", None)
    replaces = kwargs.get("replaces", None)
    replaces_if = kwargs.get("replaces_if", None)
    frame_back_times = kwargs.get("frame_back_times", 1)
    
    def _(func_or_class):
        nonlocal name
        name = func_or_class.__name__ if name is None else name
        is_class = inspect.isclass(func_or_class)
        is_method = not is_class and ismethod(func_or_class)
        module = inspect.getmodule(func_or_class)
        
        nonlocal types
        if len(types) == 0:
            assert is_method or not is_class # is method or function
            signature = inspect.signature(func_or_class)
            annotations = tuple(param_value.annotation for (param_name, param_value) in signature.parameters.items() if param_value.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            if is_method:
                annotations = annotations[1:] # throw self away
            assert all(ann is not inspect.Parameter.empty for ann in annotations)
            types = annotations
        if is_method or not is_class: # is method or function
            signature = inspect.signature(func_or_class)
            assert len(types) == len(tuple(param_name for (param_name, param_value) in signature.parameters.items() if param_value.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))) - (1 if is_method else 0) # throw self away
        
        if is_method:
            assert module_kwarg is None
            assert replaces is None
            assert replaces_if is None
            frame = inspect.currentframe()
            for _ in range(frame_back_times):
                frame = frame.f_back
            dispatcher = frame.f_locals.get(name, MethodDispatcher_Wrapper(name))
            dispatcher.add(types, func_or_class)
            return dispatcher
        else: # is function or class
            if is_class:
                assert module_kwarg is not None
                assert module_kwarg is not module # otherwise module.name is not a class anymore but a function
                module = module_kwarg
            else:
                if module_kwarg is None:
                    pass # we would like to store the dispatched function in its module, rather than the original one
                else:
                    module = module_kwarg # we would like to store the original function in its module and the dispatched in module_kwarg
            if not hasattr(module, name):
                setattr(module, name, Dispatcher(name))
            dispatcher = getattr(module, name)
            assert isinstance(dispatcher, Dispatcher)
            dispatcher.add(types, func_or_class, replaces=replaces, replaces_if=replaces_if)
            if module_kwarg is None:
                return dispatcher
            else:
                return func_or_class
    return _
dispatch.__doc__ = original_dispatch.__doc__ + \
    """
    
    This is a customized version of the @dispatch decorator provided by the
    multipledispatch package, such that an optional module kwarg is passed
    (instead of the namespace one).
        * if the object to be dispatched is a class, then
            -> @dispatch returns the original class
            -> @dispatch stores the dispatched class in module kwarg
            -> the module kwarg should be non empty
            -> the module kwarg should not contain the module in which the
               class would define, otherwise the dispatched class would be
               overwritten by the original one
          The typical case here is the implementation of a class of RBniCS
          backends
            -> we would like the return type to be the original class,
               in order to be able to use it for inheritance
            -> we would like to provide as module kwarg the rbnics.backends
               module
        * if the object to be dispatched is a class method, then
            -> @dispatch returns the dispatched method, as in the original
               implementation
            -> the module kwarg should be empty
          The typical case here is the implementation of a method of
          a class of RBniCS backends.
        * if the object to be dispatched is a function, then
            -> the module kwarg may be empty or non empty
            -> if the module kwarg is empty
                  #> @dispatch returns the dispatched function
                  #> @dispatch also stores the dispatched function in its module,
                     even though this is unnecessary as the returned dispatched
                     function would be save there
               The typical case here is the implementation of a function of
               a basic backend among RBniCS backends:
                  #> we would like the return type to be the dispatched function,
                     because we import the dispatched (basic) function in order
                     to properly implement the concrete backend
                  #> we do not mind saving the dispatched (basic) function in any
                     module, since it will not be called from any user code and
                     so it should not be saved in rbnics.backends
            -> if the module kwarg is non empty
                  #> @dispatch returns the original function
                  #> @dispatch stores the dispatched function in module kwarg
               The typical case here is the implementation of a function of
               RBniCS backends:
                  #> we would like the return type to be the original function,
                     for symmetry with the class case
                  #> we would like to provide as module kwarg the rbnics.backends
                     module
    """
    
# == Define an @overload to be used for method dispatch, in order to have a more expressive name == #
# == that hides the details of the implementation                                                == #
def overload(*args, **kwargs):
    if len(args) == 1 and inspect.isfunction(args[0]) and not islambda(args[0]) and len(kwargs) == 0:
        # called as @overload
        return dispatch(frame_back_times=2)(args[0])
    else:
        # called as @overload(*args, **kwargs)
        return dispatch(*args, **kwargs)

# == Replacements for array, dict, list, tuple that allow to specify content input types == #
class _iterable_of(object):
    def __init__(self, types):
        self.types = types
        
    def __str__(self):
        return "iterable_of(" + str(self.types) + ")"
    __repr__ = __str__

class _array_of(_iterable_of):
    def __str__(self):
        return "array_of(" + str(self.types) + ")"
    __repr__ = __str__
    
class _dict_of(object):
    def __init__(self, types_from, types_to):
        self.types_from = types_from
        self.types_to = types_to
        
    def __str__(self):
        return "dict_of(" + str(self.types_from) + ": " + str(self.types_to) + ")"
    __repr__ = __str__
    
class _list_of(_iterable_of):
    def __str__(self):
        return "list_of(" + str(self.types) + ")"
    __repr__ = __str__

class _set_of(_iterable_of):
    def __str__(self):
        return "set_of(" + str(self.types) + ")"
    __repr__ = __str__
        
class _tuple_of(_iterable_of):
    def __str__(self):
        return "tuple_of(" + str(self.types) + ")"
    __repr__ = __str__
    
_all_iterable_of_instances = dict()
_all_array_of_instances = dict()
_all_dict_of_instances = dict()
_all_list_of_instances = dict()
_all_set_of_instances = dict()
_all_tuple_of_instances = dict()

def _remove_repeated_types(types):
    if isinstance(types, tuple):
        all_types = set()
        for t in types:
            assert inspect.isclass(t) or isinstance(t, (_array_of, _dict_of, _iterable_of, _list_of, _set_of, _tuple_of))
            all_types.add(t)
        if len(all_types) == 0 or len(all_types) > 1:
            return (tuple(all_types), frozenset(all_types))
        else:
            types = all_types.pop()
            return (types, frozenset({types}))
    else:
        return (types, frozenset({types}))

def iterable_of(types):
    (types, types_set) = _remove_repeated_types(types)
    if types_set not in _all_iterable_of_instances:
        _all_iterable_of_instances[types_set] = _iterable_of(types)
    return _all_iterable_of_instances[types_set]

def array_of(types):
    (types, types_set) = _remove_repeated_types(types)
    if types_set not in _all_array_of_instances:
        _all_array_of_instances[types_set] = _array_of(types)
    return _all_array_of_instances[types_set]
    
def dict_of(types_from, types_to):
    (types_from, types_from_set) = _remove_repeated_types(types_from)
    (types_to, types_to_set) = _remove_repeated_types(types_to)
    if (types_from_set, types_to_set) not in _all_dict_of_instances:
        _all_dict_of_instances[(types_from_set, types_to_set)] = _dict_of(types_from, types_to)
    return _all_dict_of_instances[(types_from_set, types_to_set)]

def list_of(types):
    (types, types_set) = _remove_repeated_types(types)
    if types_set not in _all_list_of_instances:
        _all_list_of_instances[types_set] = _list_of(types)
    return _all_list_of_instances[types_set]
    
def set_of(types):
    (types, types_set) = _remove_repeated_types(types)
    if types_set not in _all_set_of_instances:
        _all_set_of_instances[types_set] = _set_of(types)
    return _all_set_of_instances[types_set]
    
def tuple_of(types):
    (types, types_set) = _remove_repeated_types(types)
    if types_set not in _all_tuple_of_instances:
        _all_tuple_of_instances[types_set] = _tuple_of(types)
    return _all_tuple_of_instances[types_set]
    
# == Validation that no array, dict, list, tuple are provided as types to @dispatch == #
def validate_types(inputs, allow_lambda):
    for input_ in inputs:
        type_input_ = type(input_)
        if (
            type_input_ in (list, tuple) # more strict than isinstance(input_, (list, tuple)): custom types inherited from array or list or tuple should be preserved
                or
            (type_input_ in (array, ) and input_.dtype == object)
        ):
            validate_types(input_, allow_lambda)
        else:
            assert not (input_ is array and input_.dtype == object), "Please use array_of defined in this module to specify the type of each element"
            assert input_ is not dict, "Please use dict_of defined in this module to specify the type of keys and values"
            assert input_ is not list, "Please use list_of defined in this module to specify the type of each element"
            assert input_ is not set, "Please use set_of defined in this module to specify the type of each element"
            assert input_ is not tuple, "Please use tuple_of defined in this module to specify the type of each element"
            assert (
                inspect.isclass(input_)
                    or
                isinstance(input_, (_array_of, _dict_of, _iterable_of, _list_of, _set_of, _tuple_of))
                    or
                input_ is None
                    or
                (
                    islambda(input_)
                        and
                    allow_lambda
                )
            )

# == Get types for provided inputs == #
def get_types(inputs):
    inputs = remove_trailing_None(inputs)
    types = list()
    for input_ in inputs:
        types.append(get_type(input_))
    types = tuple(types)
    return types
    
def get_type(input_):
    type_input_ = type(input_)
    if (
        type_input_ in (list, set, tuple) # more strict than isinstance(input_, (list, set, tuple)): custom types inherited from array or list or tuple should be preserved
            or
        (type_input_ in (array, ) and input_.dtype == object)
    ):
        subtypes = get_types(input_)
        subtypes = tuple(set(subtypes)) # remove repeated types
        if len(subtypes) == 1:
            subtypes = subtypes[0]
        if isinstance(input_, array):
            return array_of(subtypes)
        elif isinstance(input_, list):
            return list_of(subtypes)
        elif isinstance(input_, set):
            return set_of(subtypes)
        elif isinstance(input_, tuple):
            return tuple_of(subtypes)
        else:
            raise TypeError("Invalid type in get_types()")
    elif type_input_ in (dict, ): # more strict than isinstance(input_, (dict, ))
        subtypes_from = get_types(tuple(input_.keys()))
        subtypes_from = tuple(set(subtypes_from)) # remove repeated types
        if len(subtypes_from) == 1:
            subtypes_from = subtypes_from[0]
        subtypes_to = get_types(tuple(input_.values()))
        subtypes_to = tuple(set(subtypes_to)) # remove repeated types
        if len(subtypes_to) == 1:
            subtypes_to = subtypes_to[0]
        return dict_of(subtypes_from, subtypes_to)
    else:
        if input_ is not None:
            return type_input_
        else:
            return None
    
# == Customize tuple expansion to handle array_of, dict_of, iterable_of, list_of, set_of, tuple_of == #
def expand_tuples(L):
    if not L:
        output = {()}
    else:
        output = set((item,) + t for t in expand_tuples(L[1:]) for item in expand_arg(L[0]))
    return output
    
def expand_arg(arg, tuple_expansion=True):
    if isinstance(arg, tuple) and tuple_expansion:
        for i in arg:
            for j in expand_arg(i):
                yield j
    elif isinstance(arg, (_array_of, _list_of, _iterable_of, _set_of, _tuple_of)):
        generator_of = _generator_of[type(arg)]
        for i in powerset(arg.types):
            for j in expand_arg(i, tuple_expansion=False):
                yield generator_of(j)
    elif isinstance(arg, _dict_of):
        for i in powerset(arg.types_from):
            for j in expand_arg(i, tuple_expansion=False):
                for k in powerset(arg.types_to):
                    for l in expand_arg(k, tuple_expansion=False):
                        yield dict_of(j, l)
    else:
        yield arg
        
_generator_of = {
    _array_of: array_of,
    _dict_of: dict_of,
    _iterable_of: iterable_of,
    _list_of: list_of,
    _set_of: set_of,
    _tuple_of: tuple_of
}

# == Customize conflict operators to handle array_of, dict_of, list_of, set_of, tuple_of == #
def supercedes(A, B):
    """ A is consistent and strictly more specific than B """
    assert isinstance(A, (list, tuple))
    assert isinstance(B, (list, tuple))
    A = remove_trailing_None(A)
    B = remove_trailing_None(B)
    if len(A) != len(B):
        return False
    else:
        for (a, b) in zip(A, B):
            if isinstance(a, (_array_of, _list_of, _iterable_of, _set_of, _tuple_of)):
                if type(a) == type(b) or type(b) == _iterable_of:
                    if not supercedes((a.types, ), (b.types, )):
                        return False
                elif b is object:
                    return True
                else:
                    return False
            elif isinstance(a, _dict_of):
                if isinstance(b, _dict_of):
                    if not supercedes((a.types_from, ), (b.types_from, )) or not supercedes((a.types_to, ), (b.types_to, )):
                        return False
                elif b is object:
                    return True
                else:
                    return False
            elif isinstance(a, (list, tuple)):
                for sub_a in a:
                    if not supercedes((sub_a, ), (b, )):
                        return False
            else:
                if isinstance(b, (_array_of, _dict_of, _iterable_of, _list_of, _set_of, _tuple_of)):
                    return False
                elif isinstance(b, (list, tuple)):
                    for sub_b in b:
                        if supercedes((a, ), (sub_b, )):
                            return True
                    return False
                elif a is None and b is None:
                    continue
                elif (
                    (a is None and b is not None)
                        or
                    (b is None and a is not None)
                ):
                    return False
                elif not issubclass(a, b):
                    return False
        else:
            return True
multipledispatch.conflict.supercedes = supercedes

def consistent(A, B):
    """ It is possible for an argument list to satisfy both A and B """
    assert isinstance(A, (list, tuple))
    assert isinstance(B, (list, tuple))
    A = remove_trailing_None(A)
    B = remove_trailing_None(B)
    if len(A) != len(B):
        return False
    else:
        for (a, b) in zip(A, B):
            if isinstance(a, (_array_of, _list_of, _iterable_of, _set_of, _tuple_of)):
                if type(a) == type(b):
                    if not supercedes((a.types, ), (b.types, )) and not supercedes((b.types, ), (a.types, )):
                        return False
                else:
                    return False
            elif isinstance(a, _dict_of):
                if isinstance(b, _dict_of):
                    if (
                        (not supercedes((a.types_from, ), (b.types_from, )) or not supercedes((a.types_to, ), (b.types_to, )))
                            and
                        (not supercedes((b.types_from, ), (a.types_from, )) or not supercedes((b.types_to, ), (a.types_to, )))
                    ):
                        return False
                else:
                    return False
            else:
                if isinstance(b, (_array_of, _dict_of, _iterable_of, _list_of, _set_of, _tuple_of)):
                    return False
                elif a is None and b is None:
                    continue
                elif (
                    (a is None and b is not None)
                        or
                    (b is None and a is not None)
                ):
                    return False
                elif not issubclass(a, b) and not issubclass(b, a):
                    return False
        else:
            return True
multipledispatch.conflict.consistent = consistent

# == Helper function to remove trailing None arguments (used as default arguments) == #
def remove_trailing_None(inputs):
    assert isinstance(inputs, (array, list, set, tuple))
    if isinstance(inputs, (array, set)):
        assert None not in inputs
        return inputs
    elif isinstance(inputs, (list, tuple)):
        i = len(inputs)
        while i > 0 and inputs[i - 1] is None:
            i -= 1
        return inputs[:i]
    else:
        raise TypeError("This type is unsupported")
    
# == Helper function to create itertools.powerset without empty tuple == #
def powerset(types):
    if not isinstance(types, (list, tuple)):
        types = (types, )
    return itertools.chain.from_iterable(itertools.combinations(types, r) for r in range(1, len(types)+1))
    
# == Helper function to determine if function is a lambda function == #
def islambda(arg):
    return isinstance(arg, _reference_lambda_type) and arg.__name__ == _reference_lambda_name
_reference_lambda = lambda: 0  # noqa: E731
_reference_lambda_type = type(_reference_lambda)
_reference_lambda_name = _reference_lambda.__name__
