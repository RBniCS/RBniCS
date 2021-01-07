# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
import sys
import types
import inspect
import warnings  # noqa: F401
import urllib.request
from multipledispatch.conflict import ambiguous, ambiguities, ordering  # noqa: F401
from multipledispatch.utils import raises
from rbnics.utils.decorators import dict_of, dispatch, iterable_of, list_of, overload, tuple_of
from rbnics.utils.decorators.dispatch import (AmbiguousSignatureError, consistent, Dispatcher, InvalidSignatureError,
                                              MethodDispatcher_Wrapper as MethodDispatcher, supercedes,
                                              UnavailableSignatureError)


# Fixture to clean up current module after test execution
@pytest.fixture
def clean_main_module():
    current_state = dict(sys.modules[__name__].__dict__)
    for (name, val) in current_state.items():
        if isinstance(val, Dispatcher):
            delattr(sys.modules[__name__], name)


def apply_clean_main_module_to_tests():
    current_state = dict(sys.modules[__name__].__dict__)
    for (name, val) in current_state.items():
        if inspect.isfunction(val) and val.__name__.startswith("test_"):
            delattr(sys.modules[__name__], name)
            setattr(sys.modules[__name__], name, pytest.mark.usefixtures("clean_main_module")(val))


# Prepare tests blacklist from upstream tests
tests_blacklist = {
    "test_conflict.py": [
        "test_super_signature", "test_type_mro",
        # we are not interested in the super signature warning
        "test_supercedes_variadic", "test_consistent_variadic",
        # variadic arguments are not supported by our custom version
    ],
    "test_core.py": [
        "test_competing_ambiguous",
        # patched version raises an error rather than a warning and this would stop the execution
        "test_namespaces",
        # we have replaced namespaces with modules so it makes no sense to test this
        "test_multipledispatch",
        # has issues with recent pytest versions
    ],
    "test_dispatcher.py": [
        "test_register_instance_method", "test_register_stacking", "test_dispatch_method",
        # created a custom version of that replaces unallowed list with list_of(int) and tuple_of(int)
        "test_source",
        # disabled because of incompatibility with exec (failure with OSError)
        "test_not_implemented", "test_not_implemented_error",
        # handling of MDNotImplementedError has been disabled by us
        "test_vararg_not_last_element_of_signature", "test_vararg_has_multiple_elements",
        "test_vararg_dispatch_simple", "test_vararg_dispatch_ambiguity",
        "test_vararg_dispatch_ambiguity_in_variadic",
        "test_vararg_dispatch_multiple_types_explicit_args",
        "test_vararg_dispatch_multiple_implementations",
        "test_vararg_dispatch_unions", "test_vararg_no_args",
        "test_vararg_no_args_failure", "test_vararg_no_args_failure_2", "test_vararg_ordering",
        # variadic arguments are not supported by our custom version
    ]
}


# Prepare additional test renames
test_renames = {
    "test_conflict.py": {},
    "test_core.py": {
        "test_union_types": "test_union_types_core"
    },
    "test_dispatcher.py": {
        "test_union_types": "test_union_types_dispatcher"
    }
}


# Get upstream tests from github
def download_multipledispatch_tests(file_name, start_from_line):
    response = urllib.request.urlopen(
        "https://raw.githubusercontent.com/mrocklin/multipledispatch/master/multipledispatch/tests/" + file_name)
    lines = response.read().decode("utf-8").split("\n")[start_from_line:]
    # Get all classes which have been defined
    classes = ""
    for line in lines:
        if line.startswith("class "):
            classes += "    " + line + "\n"
    # Make class definition local to each function
    code = ""
    for line in lines:
        if line.startswith("class "):
            pass
        elif line.startswith("def "):
            code += line + "\n"
            code += classes + "\n"
        else:
            code += line + "\n"
    # Blacklist tests that should not be run
    for test in tests_blacklist[file_name]:
        code = code.replace("def " + test + "(", "def _disabled_" + test + "(")
    # Rename tests
    for (test_original_name, test_new_name) in test_renames[file_name].items():
        code = code.replace("def " + test_original_name + "(", "def " + test_new_name + "(")
    return code


exec(download_multipledispatch_tests("test_conflict.py", 4))
exec(download_multipledispatch_tests("test_core.py", 10))
exec(download_multipledispatch_tests("test_dispatcher.py", 7))


# Add back patched version of ambiguous test of test_core.py
def test_competing_ambiguous_try():

    class A(object):
        pass

    class C(A):
        pass

    @dispatch(A, C)
    def f(x, y):
        return 2

    @dispatch(C, A)
    def f(x, y):
        return 2

    assert f(A(), C()) == f(C(), A()) == 2

    try:
        f(C(), C())
    except AmbiguousSignatureError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("This test should have raised an AmbiguousSignatureError")


# Add back patched versions of list tests of test_dispatcher.py
def test_register_instance_method__list_of():

    class Test(object):
        __init__ = MethodDispatcher("f")

        @__init__.register(list_of(int))
        def _init_list(self, data):
            self.data = data

        @__init__.register(object)
        def _init_obj(self, datum):
            self.data = [datum]

    a = Test(3)
    b = Test([3])
    assert a.data == b.data


def test_register_stacking__list_of__tuple_of():
    f = Dispatcher("f")

    @f.register(list_of(int))
    @f.register(tuple_of(int))
    def rev(x):
        return x[::-1]

    assert f((1, 2, 3)) == (3, 2, 1)
    assert f([1, 2, 3]) == [3, 2, 1]

    assert raises(UnavailableSignatureError, lambda: f("hello"))
    assert rev("hello") == "olleh"


def test_dispatch_method__list_of():
    f = Dispatcher("f")

    @f.register(list_of(int))
    def rev(x):
        return x[::-1]

    @f.register(int, int)
    def add(x, y):
        return x + y

    class MyList(list):
        pass

    assert f.dispatch(list_of(int)) is rev
    assert f.dispatch(MyList) is None  # MyList is considered by list_of() as an independent type
    assert f.dispatch(int, int) is add


# Add back the same version of test_source from test_dispatcher, which is incompatible with exec()
def test_source_copy():

    def one(x, y):
        """ Docstring number one """
        return x + y

    def two(x, y):
        """ Docstring number two """
        return x - y

    master_doc = "Doc of the multimethod itself"

    f = Dispatcher("f", doc=master_doc)
    f.add((int, int), one)
    f.add((float, float), two)

    assert "x + y" in f._source(1, 1)
    assert "x - y" in f._source(1.0, 1.0)


# Add back subset of upstream test_dispatcher_3only.py, which is not tested fully because of
# our different implementation of dispatching through annotations
def test_function_annotation_dispatch():

    @dispatch()
    def inc2(x: int):
        return x + 1

    @dispatch()
    def inc2(x: float):
        return x - 1

    assert inc2(1) == 2
    assert inc2(1.0) == 0.0


def test_method_annotations():

    class Foo(object):
        @dispatch()
        def f(self, x: int):
            return x + 1

        @dispatch()
        def f(self, x: float):
            return x - 1

    foo = Foo()

    assert foo.f(1) == 2
    assert foo.f(1.0) == 0.0


# Test on supercedes
def test_supercedes_2():

    class A(object):
        pass

    class B(A):
        pass

    class C(object):
        pass

    assert supercedes((B, ), (A, ))
    assert supercedes((B, A), (A, A))
    assert not supercedes((B, A), (A, B))
    assert not supercedes((A, ), (B, ))
    assert supercedes((B, ), ((A, C), ))
    assert supercedes((C, ), ((A, C), ))
    assert not supercedes((A, ), ((B, C), ))
    assert not supercedes(((A, B), ), (B, ))
    assert supercedes(((A, B), ), (A, ))
    assert supercedes((A, ), ((A, B), ))
    assert supercedes((B, ), ((A, B), ))

    assert supercedes((iterable_of(B), ), (iterable_of(A), ))
    assert supercedes((iterable_of((B, A)), ), (iterable_of(A), ))
    assert supercedes((iterable_of((B, A)), ), (iterable_of((A, A)), ))
    assert supercedes((iterable_of((B, A)), ), (iterable_of((A, B)), ))
    assert supercedes((iterable_of((B, B)), ), (iterable_of((B)), ))
    assert supercedes((iterable_of((B, B)), ), (iterable_of((A, B)), ))
    assert supercedes((iterable_of(B), B), (iterable_of(A), A))
    assert not supercedes((iterable_of(A), ), (iterable_of(B), ))

    assert supercedes((list_of(B), ), (list_of(A), ))
    assert supercedes((list_of((B, A)), ), (list_of(A), ))
    assert supercedes((list_of((B, A)), ), (list_of((A, A)), ))
    assert supercedes((list_of((B, A)), ), (list_of((A, B)), ))
    assert supercedes((list_of((B, B)), ), (list_of((B)), ))
    assert supercedes((list_of((B, B)), ), (list_of((A, B)), ))
    assert supercedes((list_of(B), B), (list_of(A), A))
    assert not supercedes((list_of(A), ), (list_of(B), ))

    assert supercedes((list_of(B), ), (iterable_of(A), ))
    assert not supercedes((iterable_of(B), ), (list_of(A), ))
    assert supercedes((list_of((B, A)), ), (iterable_of(A), ))
    assert not supercedes((iterable_of((B, A)), ), (list_of(A), ))
    assert supercedes((list_of((B, A)), ), (iterable_of((A, A)), ))
    assert not supercedes((iterable_of((B, A)), ), (list_of((A, A)), ))
    assert supercedes((list_of((B, A)), ), (iterable_of((A, B)), ))
    assert not supercedes((iterable_of((B, A)), ), (list_of((A, B)), ))
    assert supercedes((list_of((B, B)), ), (iterable_of((B)), ))
    assert not supercedes((iterable_of((B, B)), ), (list_of((B)), ))
    assert supercedes((list_of((B, B)), ), (iterable_of((A, B)), ))
    assert not supercedes((iterable_of((B, B)), ), (list_of((A, B)), ))
    assert supercedes((list_of(B), B), (iterable_of(A), A))
    assert not supercedes((iterable_of(B), B), (list_of(A), A))
    assert not supercedes((list_of(A), ), (iterable_of(B), ))
    assert not supercedes((iterable_of(A), ), (list_of(B), ))

    assert supercedes((dict_of(B, C), ), (dict_of(A, C), ))
    assert supercedes((dict_of(B, C), ), (dict_of((A, C), C), ))
    assert supercedes((dict_of(C, C), ), (dict_of((A, C), C), ))
    assert not supercedes((dict_of((B, C), C), ), (dict_of(A, C), ))
    assert supercedes((dict_of(list_of(B), C), ), (dict_of(list_of(A), C), ))


# Test on consistent
def test_consistent_2():

    class A(object):
        pass

    class B(A):
        pass

    class C(object):
        pass

    assert consistent((A, ), (A, ))
    assert consistent((B, ), (B, ))
    assert not consistent((A, ), (C, ))
    assert consistent((A, B), (A, B))
    assert consistent((B, A), (A, B))
    assert not consistent((B, A), (B, ))
    assert not consistent((B, A), (B, C))

    assert consistent((iterable_of(A), ), (iterable_of(A), ))
    assert consistent((iterable_of(B), ), (iterable_of(B), ))
    assert not consistent((iterable_of(A), ), (iterable_of(C), ))
    assert consistent((iterable_of((A, B)), ), (iterable_of((A, B)), ))
    assert consistent((iterable_of((B, A)), ), (iterable_of((A, B)), ))
    assert consistent((iterable_of((B, A)), ), (iterable_of(B), ))
    assert consistent((iterable_of((B, A)), ), (iterable_of(A), ))
    assert not consistent((iterable_of((B, A)), ), (iterable_of((B, C)), ))

    assert consistent((list_of(A), ), (list_of(A), ))
    assert consistent((list_of(B), ), (list_of(B), ))
    assert not consistent((list_of(A), ), (list_of(C), ))
    assert consistent((list_of((A, B)), ), (list_of((A, B)), ))
    assert consistent((list_of((B, A)), ), (list_of((A, B)), ))
    assert consistent((list_of((B, A)), ), (list_of(B), ))
    assert consistent((list_of((B, A)), ), (list_of(A), ))
    assert not consistent((list_of((B, A)), ), (list_of((B, C)), ))

    assert not consistent((list_of(A), ), (iterable_of(A), ))
    assert not consistent((iterable_of(A), ), (list_of(A), ))
    assert not consistent((list_of(B), ), (iterable_of(B), ))
    assert not consistent((iterable_of(B), ), (list_of(B), ))
    assert not consistent((list_of(A), ), (iterable_of(C), ))
    assert not consistent((iterable_of(A), ), (list_of(C), ))
    assert not consistent((list_of((A, B)), ), (iterable_of((A, B)), ))
    assert not consistent((iterable_of((A, B)), ), (list_of((A, B)), ))
    assert not consistent((list_of((B, A)), ), (iterable_of((A, B)), ))
    assert not consistent((iterable_of((B, A)), ), (list_of((A, B)), ))
    assert not consistent((list_of((B, A)), ), (iterable_of(B), ))
    assert not consistent((iterable_of((B, A)), ), (list_of(B), ))
    assert not consistent((list_of((B, A)), ), (iterable_of(A), ))
    assert not consistent((iterable_of((B, A)), ), (list_of(A), ))
    assert not consistent((list_of((B, A)), ), (iterable_of((B, C)), ))
    assert not consistent((iterable_of((B, A)), ), (list_of((B, C)), ))

    assert consistent((dict_of(A, C), ), (dict_of(A, C), ))
    assert consistent((dict_of(B, C), ), (dict_of(B, C), ))
    assert not consistent((dict_of(A, C), ), (dict_of(C, C), ))


# Test on ambiguous
def test_ambiguous_2():

    class A(object):
        pass

    class B(A):
        pass

    class C(object):
        pass

    assert not ambiguous((A, ), (A, ))
    assert not ambiguous((A, ), (B, ))
    assert not ambiguous((B, ), (B, ))
    assert not ambiguous((A, B), (B, B))
    assert ambiguous((A, B), (B, A))

    assert not ambiguous((list_of(A), ), (list_of(A), ))
    assert not ambiguous((list_of(B), ), (list_of(B), ))
    assert not ambiguous((list_of(A), ), (list_of(C), ))
    assert not ambiguous((list_of((A, B)), ), (list_of((A, B)), ))
    assert not ambiguous((list_of((B, A)), ), (list_of((A, B)), ))
    assert not ambiguous((list_of((B, A)), ), (list_of(B), ))
    assert not ambiguous((list_of((B, A)), ), (list_of(A), ))
    assert not ambiguous((list_of((B, A)), ), (list_of((B, C)), ))


# Test function with dispatch to custom module
def test_function_custom_module():
    module = types.ModuleType("module", "A runtime generated module")

    @dispatch(int, module=module)
    def f(arg):
        return arg

    assert inspect.isfunction(f)
    assert isinstance(module.f, Dispatcher)

    @dispatch(float, module=module)
    def f(arg):
        return 2. * arg

    assert inspect.isfunction(f)
    assert isinstance(module.f, Dispatcher)

    @dispatch(str, module=module)
    def f(arg):
        return arg[0]

    assert inspect.isfunction(f)
    assert isinstance(module.f, Dispatcher)

    f = module.f
    assert f(1) == 1
    assert f(1.) == 2.
    assert f("test") == "t"
    assert raises(UnavailableSignatureError, lambda: f(object()))


# Test function with dispatch to current module
def test_function_current_module():

    @dispatch(int)
    def f(arg):
        return arg

    assert isinstance(f, Dispatcher)

    @dispatch(float)
    def f(arg):
        return 2. * arg

    assert isinstance(f, Dispatcher)

    @dispatch(str)
    def f(arg):
        return arg[0]

    assert isinstance(f, Dispatcher)

    assert f(1) == 1
    assert f(1.) == 2.
    assert f("test") == "t"
    assert raises(UnavailableSignatureError, lambda: f(object()))


# Dispatch based on annotations for functions
def test_dispatch_annotation():

    @dispatch()
    def f(arg: int) -> int:
        return arg

    assert f(1) == 1


# Try to dispatch twice for the same argument type for function
def test_dispatch_twice_function():

    @dispatch(int)
    def g(arg):
        return arg

    try:
        @dispatch(int)
        def g(arg):
            return 10 * arg
    except AmbiguousSignatureError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("Duplicate @dispatch() should have failed")


# Try to implicitly solve twice dispatch with tuples
def test_dispatch_twice_function_automatic_resolution():

    @dispatch(float)
    def g(arg):
        return arg

    assert raises(UnavailableSignatureError, lambda: g(1))
    assert g(1.) == 1.

    @dispatch((int, float))
    def g(arg):
        return 2 * arg

    assert g(1) == 2
    assert g(1.) == 1.

    @dispatch(int)
    def g(arg):
        return 3 * arg

    assert g(1) == 3
    assert g(1.) == 1.


# Try to replace a function
def test_dispatch_replace_function():
    module = types.ModuleType("module", "A runtime generated module")

    @dispatch(int, module=module)
    def g(arg):
        return arg

    g_int_1 = g
    assert inspect.isfunction(g_int_1)

    @dispatch(int, module=module, replaces=g_int_1)
    def g(arg):
        return 2 * arg

    g_int_2 = g
    assert inspect.isfunction(g_int_2)
    assert module.g(1) == 2

    try:
        @dispatch(int, module=module, replaces=g_int_1)
        def g(arg):
            return 3 * arg
    except AssertionError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("Incorrect replacing in @dispatch() should have failed")

    @dispatch(int, module=module, replaces=g_int_2)
    def g(arg):
        return 4 * arg

    g_int_4 = g
    assert inspect.isfunction(g_int_4)
    assert module.g(1) == 4

    @dispatch(int, module=module, replaces=g_int_4, replaces_if=lambda arg: arg % 2 == 0)
    def g(arg):
        return 5 * arg

    g_int_5 = g
    assert inspect.isfunction(g_int_5)
    assert module.g(1) == 4
    assert module.g(2) == 10


# Test class dispatch
def test_class_custom_module():
    module = types.ModuleType("module", "A runtime generated module")

    @dispatch(int, module=module)
    class A(object):
        def __init__(self, arg):
            self.arg = arg

    assert inspect.isclass(A)
    assert isinstance(module.A, Dispatcher)

    @dispatch(float, module=module)
    class A(object):
        def __init__(self, arg):
            self.arg = 2. * arg

    assert inspect.isclass(A)
    assert isinstance(module.A, Dispatcher)

    @dispatch(str, module=module)
    class A(object):
        def __init__(self, arg):
            self.arg = arg[0]

    assert inspect.isclass(A)
    assert isinstance(module.A, Dispatcher)

    A = module.A
    assert A(1).arg == 1
    assert A(1.).arg == 2.
    assert A("test").arg == "t"


# Test methods, with both types list and annotations
def test_methods_2():

    class B(object):
        def __init__(self, arg):
            self.arg = arg

        @dispatch(int)
        def __mul__(self, other):
            return self.arg * other

        @dispatch(float)
        def __mul__(self, other):
            return 2. * self.arg * other

        @dispatch()
        def __mul__(self, other: str):
            return other[0] + "*" + str(self.arg)

    assert B(1) * 2 == 2
    assert B(1) * 2. == 4.
    assert B(1) * "test" == "t*1"


# Test class with list inputs
def test_disable_list():
    try:
        class D(object):
            @dispatch(list)
            def __init__(self, arg):
                self.arg = arg[0]
    except InvalidSignatureError:
        # this has failed has expected, because list is not supposed to be used, and list_of should be used instead
        pass
    else:
        raise RuntimeError("@dispatch(list) should have failed")


# Test class with list_of inputs
def test_enabled_list_of():

    class E(object):
        @dispatch(list_of(int))
        def __init__(self, arg):
            self.arg = arg[0]

    assert E([1, 2]).arg == 1


# Test class with list_of inputs, with multiple types for each list
def test_enabled_list_of_multiple():

    class E(object):
        @dispatch(list_of(float))
        def __init__(self, arg):
            self.arg = arg[0]

        @dispatch(list_of((float, int)))
        def __init__(self, arg):
            self.arg = 2 * arg[0]

        @dispatch(list_of((float, str)))
        def __init__(self, arg):
            self.arg = 3 * arg[0]

    assert E([1.]).arg == 1. and isinstance(E([1., 2.]).arg, float)
    assert E(["a"]).arg == "aaa"
    assert E([1]).arg == 2 and isinstance(E([1]).arg, int)
    assert E([1., 2.]).arg == 1. and isinstance(E([1., 2.]).arg, float)
    assert E([4., 3]).arg == 8. and isinstance(E([4., 3]).arg, float)
    assert E([4, 3.]).arg == 8 and isinstance(E([4, 3.]).arg, int)
    assert E([2., "a"]).arg == 6. and isinstance(E([2., "a"]).arg, float)
    assert raises(UnavailableSignatureError, lambda: E([object()]))


# Test class with list_of inputs, with multiple types for each list, with ambiguous types
def test_enabled_list_of_multiple_ambiguous():

    class E(object):
        @dispatch(list_of(float))
        def __init__(self, arg):
            self.arg = arg[0]

        @dispatch(list_of((float, int)))
        def __init__(self, arg):
            self.arg = 2 * arg[0]

        @dispatch(list_of((float, int, str)))
        def __init__(self, arg):
            self.arg = 3 * arg[0]

    try:
        # Ambiguity checking is delayed to the first time the method is evaluated
        E = E(None)
        # The ambiguity for list_of(float) is solved by the first method, however
        # the ambiguity for list_of(int) cannot be solved as both the second and third
        # method could be used for that.
    except AmbiguousSignatureError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("This test should have failed due to list_of(int) ambiguity")

    # Ambiguity can be fixed by explicitly providing an overloaded method for list_of(int)
    class F(object):
        @dispatch(list_of(float))
        def __init__(self, arg):
            self.arg = arg[0]

        @dispatch(list_of(int))
        def __init__(self, arg):
            self.arg = 4 * arg[0]

        @dispatch(list_of((float, int)))
        def __init__(self, arg):
            self.arg = 2 * arg[0]

        @dispatch(list_of((float, int, str)))
        def __init__(self, arg):
            self.arg = 3 * arg[0]

    assert F([1.]).arg == 1. and isinstance(F([1., 2.]).arg, float)
    assert F(["a"]).arg == "aaa"
    assert F([1]).arg == 4 and isinstance(F([1]).arg, int)
    assert F([1., 2.]).arg == 1. and isinstance(F([1., 2.]).arg, float)
    assert F([4., 3]).arg == 8. and isinstance(F([4., 3]).arg, float)
    assert F([4, 3.]).arg == 8 and isinstance(F([4, 3.]).arg, int)
    assert F([2., "a"]).arg == 6. and isinstance(F([2., "a"]).arg, float)


# Try to dispatch twice on the same argument type for class
def test_dispatch_twice_class():
    module = types.ModuleType("module", "A runtime generated module")

    @dispatch(int, module=module)
    class G(object):
        def __init__(self, arg):
            self.arg = arg

    assert inspect.isclass(G)
    assert isinstance(module.G, Dispatcher)

    try:
        @dispatch(int, module=module)
        class G(object):
            def __init__(self, arg):
                self.arg = 10 * arg
    except AmbiguousSignatureError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("Duplicate @dispatch() should have failed")


# Try to replace a class
def test_dispatch_replace_class():
    module = types.ModuleType("module", "A runtime generated module")

    @dispatch(int, module=module)
    class G(object):
        def __init__(self, arg):
            self.arg = arg

    G_int_1 = G
    assert inspect.isclass(G_int_1)

    @dispatch(int, module=module, replaces=G_int_1)
    class G(object):
        def __init__(self, arg):
            self.arg = 2 * arg

    G_int_2 = G
    assert inspect.isclass(G_int_2)
    assert module.G(1).arg == 2

    try:
        @dispatch(int, module=module, replaces=G_int_1)
        class G(object):
            def __init__(self, arg):
                self.arg = 3 * arg
    except AssertionError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("Incorrect replacing in @dispatch() should have failed")

    @dispatch(int, module=module, replaces=G_int_2)
    class G(object):
        def __init__(self, arg):
            self.arg = 4 * arg

    G_int_4 = G
    assert inspect.isclass(G_int_4)
    assert module.G(1).arg == 4

    @dispatch(int, module=module, replaces=G_int_4, replaces_if=lambda arg: arg % 2 == 0)
    class G(object):
        def __init__(self, arg):
            self.arg = 5 * arg

    G_int_5 = G
    assert inspect.isclass(G_int_5)
    assert module.G(1).arg == 4
    assert module.G(2).arg == 10


# Try to dispatch twice on the same argument type for method
def test_dispatch_twice_method():
    try:
        class H(object):
            @dispatch(int)
            def __mul__(self, other):
                return other

            @dispatch(int)
            def __mul__(self, other):
                return 10 * other
    except AmbiguousSignatureError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("Duplicate @dispatch() should have failed")


# Test inheritance in combination with list_of
def test_inheritance_for_list_of():

    class A(object):
        pass

    class B(object):
        pass

    class C(A):
        pass

    @dispatch(list_of(A))
    def f(x):
        return "a"

    @dispatch(list_of(B))
    def f(x):
        return "b"

    assert f([A(), A()]) == "a"
    assert f([B(), B()]) == "b"
    assert f([C(), C()]) == "a"
    assert f([C(), A()]) == "a"
    assert raises(UnavailableSignatureError, lambda: f(B(), C()))


# Test inheritance in combination with dict_of (keys only)
def test_inheritance_for_dict_of_keys():

    class A(object):
        pass

    class B(object):
        pass

    class C(A):
        pass

    @dispatch(dict_of(A, int))
    def f(x):
        return "a"

    @dispatch(dict_of(B, int))
    def f(x):
        return "b"

    assert f({A(): 1}) == "a"
    assert f({B(): 2}) == "b"
    assert f({C(): 3}) == "a"
    assert raises(UnavailableSignatureError, lambda: f({B(): 4.}))


# Test inheritance in combination with dict_of (values only)
def test_inheritance_for_dict_of_values():

    class A(object):
        pass

    class B(object):
        pass

    class C(A):
        pass

    @dispatch(dict_of(int, A))
    def f(x):
        return "a"

    @dispatch(dict_of(int, B))
    def f(x):
        return "b"

    assert f({1: A()}) == "a"
    assert f({2: B()}) == "b"
    assert f({3: C()}) == "a"
    assert raises(UnavailableSignatureError, lambda: f({4.: B()}))


# Test inheritance in combination with dict_of (keys and values)
def test_inheritance_for_dict_of_keys_values():

    class A(object):
        pass

    class B(object):
        pass

    class C(A):
        pass

    @dispatch(dict_of(A, A))
    def f(x):
        return "a"

    @dispatch(dict_of(B, B))
    def f(x):
        return "b"

    assert f({A(): A()}) == "a"
    assert f({B(): B()}) == "b"
    assert f({C(): C()}) == "a"
    assert f({A(): C()}) == "a"
    assert f({C(): A()}) == "a"
    assert raises(UnavailableSignatureError, lambda: f({A(): B()}))


# Test inheritance in combination with dict_of (keys only, which are tuple_of)
def test_inheritance_for_dict_of_keys_tuple_of():

    class A(object):
        pass

    class B(object):
        pass

    class C(A):
        pass

    @dispatch(dict_of(tuple_of(A), int))
    def f(x):
        return "a"

    @dispatch(dict_of(tuple_of(B), int))
    def f(x):
        return "b"

    assert f({(A(), A()): 1}) == "a"
    assert f({(B(), B()): 2}) == "b"
    assert f({(C(), C()): 3}) == "a"
    assert f({(C(), A()): 4}) == "a"
    assert raises(UnavailableSignatureError, lambda: f({(B(), B()): 5.}))


# Test competing solutions for list_of
def test_competing_solutions_for_list_of():

    class A(object):
        pass

    class C(A):
        pass

    class D(C):
        pass

    @dispatch(list_of(A))
    def h(x):
        return 1

    @dispatch(list_of(C))
    def h(x):
        return 2

    assert h([A()]) == 1
    assert h([C()]) == 2
    assert h([D()]) == 2


# Test competing solutions for dict_of (keys only)
def test_competing_solutions_for_dict_of_keys():

    class A(object):
        pass

    class C(A):
        pass

    class D(C):
        pass

    @dispatch(dict_of(A, int))
    def h(x):
        return 1

    @dispatch(dict_of(C, int))
    def h(x):
        return 2

    assert h({A(): 1}) == 1
    assert h({C(): 2}) == 2
    assert h({D(): 3}) == 2
    assert raises(UnavailableSignatureError, lambda: h({A(): 4.}))


# Test competing solutions for dict_of (values only)
def test_competing_solutions_for_dict_of_values():

    class A(object):
        pass

    class C(A):
        pass

    class D(C):
        pass

    @dispatch(dict_of(int, A))
    def h(x):
        return 1

    @dispatch(dict_of(int, C))
    def h(x):
        return 2

    assert h({1: A()}) == 1
    assert h({2: C()}) == 2
    assert h({3: D()}) == 2
    assert raises(UnavailableSignatureError, lambda: h({4.: A()}))


# Test competing solutions for dict_of (keys and values)
def test_competing_solutions_for_dict_of_keys_and_values():

    class A(object):
        pass

    class C(A):
        pass

    class D(C):
        pass

    @dispatch(dict_of(A, A))
    def h(x):
        return 1

    @dispatch(dict_of(C, C))
    def h(x):
        return 2

    assert h({A(): A()}) == 1
    assert h({C(): C()}) == 2
    assert h({D(): D()}) == 2
    assert h({A(): D()}) == 1
    assert h({D(): A()}) == 1
    assert raises(UnavailableSignatureError, lambda: h({A(): object()}))


# Test that cache is cleared also for list_of
def test_caching_correct_behavior_list_of():

    class A(object):
        pass

    class C(A):
        pass

    @dispatch(list_of(A))
    def f(x):
        return 1

    assert f([C()]) == 1

    @dispatch(list_of(C))
    def f(x):
        return 2

    assert f([C()]) == 2


# Test union types where one of them is a list_of
def test_union_types_list_of():

    class A(object):
        pass

    class C(A):
        pass

    @dispatch((A, list_of(C)))
    def f(x):
        return 1

    assert f(A()) == 1
    assert f([C()]) == 1


# Test class with override from parent class
def test_override():

    class B(object):
        def __init__(self, arg):
            self.arg = arg

        @dispatch(int)
        def __mul__(self, other):
            return self.arg * other

        @dispatch(float)
        def __mul__(self, other):
            return 2. * self.arg * other

    class C(B):
        @dispatch(str)
        def __mul__(self, other):
            return 3 * self.arg * other

    assert B(1) * 2 == 2
    assert B(1) * 2. == 4.
    assert raises(UnavailableSignatureError, lambda: B(1) * "test")
    assert C(1) * 2 == 2
    assert C(1) * 2. == 4.
    assert C(1) * "test" == "testtesttest"


# Test class with override from parent class (when parent and child have a method with the same signature)
def test_override_2():

    class B(object):
        def __init__(self, arg):
            self.arg = arg

        @dispatch(int)
        def __mul__(self, other):
            return self.arg * other

        @dispatch(float)
        def __mul__(self, other):
            return 2. * self.arg * other

    class C(B):
        @dispatch(str)
        def __mul__(self, other):
            return 3 * self.arg * other

        @dispatch(int)
        def __mul__(self, other):
            return 4. * self.arg * int(other)

    assert B(1) * 2 == 2
    assert B(1) * 2. == 4.
    assert raises(UnavailableSignatureError, lambda: B(1) * "test")
    assert C(1) * 2 == 8  # and not 2
    assert C(1) * 2. == 4.
    assert C(1) * "test" == "testtesttest"


# Test class with override from parent class (when parent class had a non dispatched method)
def test_override_3():

    class B(object):
        def __init__(self, arg):
            self.arg = arg

        def __mul__(self, other):
            return self.arg * other

    class C(B):
        @dispatch(int)
        def __mul__(self, other):
            return 2. * self.arg * int(other)

    assert B(1) * 2 == 2
    assert B(1) * 2. == 2.
    assert C(1) * 2 == 4  # and not 2
    assert C(1) * 2. == 2.


# Test class with overrides with two levels of inheritance
def test_override_4():

    class B(object):
        def __init__(self, arg):
            self.arg = arg

        @dispatch(int)
        def __mul__(self, other):
            return self.arg * other

    class C(B):
        @dispatch(float)
        def __mul__(self, other):
            return 2. * self.arg * other

    class D(C):
        @dispatch(str)
        def __mul__(self, other):
            return 3 * self.arg * other

    assert B(1) * 2 == 2
    assert raises(UnavailableSignatureError, lambda: B(1) * 2.)
    assert raises(UnavailableSignatureError, lambda: B(1) * "test")
    assert C(1) * 2 == 2
    assert C(1) * 2. == 4.
    assert raises(UnavailableSignatureError, lambda: C(1) * "test")
    assert D(1) * 2 == 2
    assert D(1) * 2. == 4.
    assert D(1) * "test" == "testtesttest"


# Test class with lambda inputs
def test_method_lambda_function():

    class F(object):
        Type1 = int
        Type2 = float
        Type3 = str

        def __init__(self, arg):
            self.arg = arg

        @dispatch(lambda cls: cls.Type1)
        def __mul__(self, other):
            return self.arg * other

        @dispatch(lambda cls: cls.Type2)
        def __mul__(self, other):
            return 2. * self.arg * other

        @dispatch(lambda cls: cls.Type3)
        def __mul__(self, other):
            return other[0] + "*" + str(self.arg)

    assert F(1) * 2 == 2
    assert F(1) * 2. == 4.
    assert F(1) * "test" == "t*1"


# Test class with lambda inputs, changing by mistake the underlying type
def test_method_lambda_function_change():

    class F(object):
        Type = int

        def __init__(self, arg):
            self.arg = arg

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return self.arg * other

    assert F(3) * 2 == 6 and isinstance(F(3) * 2, int)
    F.Type = float                                               # this has no effect because
    assert raises(UnavailableSignatureError, lambda: F(3) * 2.)  # the Type has been already processed,
    assert F(3) * 2 == 6 and isinstance(F(3) * 2, int)           # so the internal signature is unchanged


# Test class with lambda inputs and inheritance which does not override a method,
# but changes the return value of the lambda function
def test_method_lambda_function_no_overrides():

    class F(object):
        Type = int

        def __init__(self, arg):
            self.arg = arg

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return self.arg * other

    class G(F):
        Type = float

    class H(F):
        Type = str

    assert F(1) * 2 == 2
    assert raises(UnavailableSignatureError, lambda: G(1) * 2)
    assert G(2) * 2. == 4.
    assert raises(UnavailableSignatureError, lambda: G(3) * "test")
    assert raises(UnavailableSignatureError, lambda: H(1) * 2)
    assert raises(UnavailableSignatureError, lambda: H(2) * 2.)
    assert H(3) * "test" == "testtesttest"


# Test class with lambda inputs and overrides of a non-dispatched parent method
def test_method_lambda_function_overrides_non_dispatched_parent_method():

    class A(object):
        def __init__(self, arg):
            self.arg = arg

        def __mul__(self, other):
            return self.arg * other

    class B(A):
        Type = int

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return 2 * self.arg * other

    class C(A):
        Type = object

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return 2 * self.arg * other

    assert A(1) * 2 == 2
    assert A(3) * "test" == "testtesttest"
    assert B(1) * 2 == 4
    assert B(3) * "test" == "testtesttest"
    assert C(1) * 2 == 4
    assert C(3) * "test" == "testtesttesttesttesttest"


# Test class with lambda inputs and overrides of a non-lambda parent dispatched method
def test_method_lambda_function_overrides_non_lambda_parent_dispatched_method():

    class A(object):
        def __init__(self, arg):
            self.arg = arg

        @dispatch(int)
        def __mul__(self, other):
            return self.arg * other

    class B(A):
        Type = int

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return 2 * self.arg * other

    assert A(1) * 2 == 2
    assert B(1) * 2 == 4


# Test class with lambda inputs which is overriden by a non-lambda parent dispatched method
def test_method_lambda_function_overriden_by_non_lambda_parent_dispatched_method():

    class A(object):
        Type = int

        def __init__(self, arg):
            self.arg = arg

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return self.arg * other

    class B(A):
        @dispatch(int)
        def __mul__(self, other):
            return 2 * self.arg * other

    assert A(1) * 2 == 2
    assert B(1) * 2 == 4


# Test class with lambda inputs and overrides of a lambda parent dispatched method
def test_method_lambda_function_overrides_lambda_parent_dispatched_method():

    class A(object):
        Type = int

        def __init__(self, arg):
            self.arg = arg

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return self.arg * other

    class B(A):
        Type = int

        @dispatch(lambda cls: cls.Type)
        def __mul__(self, other):
            return 2 * self.arg * other

    assert A(1) * 2 == 2
    assert B(1) * 2 == 4


# Test overload decorator
def test_overload_decorator():

    class F(object):
        Type3 = str

        def __init__(self, arg):
            self.arg = arg

        @overload(int)
        def __mul__(self, other):
            return self.arg * other

        @overload
        def __mul__(self, other: float):
            return 2. * self.arg * other

        @overload(lambda cls: cls.Type3)
        def __mul__(self, other):
            return other[0] + "*" + str(self.arg)

    assert F(1) * 2 == 2
    assert F(1) * 2. == 4.
    assert F(1) * "test" == "t*1"


# Test None arguments
def test_None_arguments():

    @dispatch(None)
    def f(x=None):
        return "None1"

    @dispatch(int, None)
    def f(x, y=None):
        return str(x) + str(y) + "2"

    @dispatch(None, int)
    def f(x, y):
        return str(x) + str(y) + "3"

    @dispatch(int, int)
    def f(x, y):
        return str(x) + str(y) + "4"

    assert f() == "None1"
    assert f(None) == "None1"
    assert f(1) == "1None2"
    assert f(1, None) == "1None2"
    assert f(None, 1) == "None13"
    assert f(1, 2) == "124"


# Test wrong number of arguments
def test_wrong_number_of_arguments():

    try:
        @dispatch()
        def f(a):
            return a
    except AssertionError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("This test should have failed due to wrong number of arguments")

    try:
        @dispatch(int)
        def f(a, b):
            return a + b
    except AssertionError:
        pass  # this has failed has expected
    else:
        raise RuntimeError("This test should have failed due to wrong number of arguments")


# Test calling method from class rather than instance
def test_call_method_from_class():

    class A(object):
        @dispatch(int)
        def show(self, number):
            return number

    a = A()
    assert a.show(1) == 1
    assert A.show(a, 1) == 1


# Apply fixture to clean up current module after test execution
apply_clean_main_module_to_tests()
