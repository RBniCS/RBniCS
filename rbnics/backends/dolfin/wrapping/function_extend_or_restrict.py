# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from collections import OrderedDict
from dolfin import assign, Function


def function_extend_or_restrict(function, function_components, V, V_components, weight, copy,
                                extended_or_restricted_function=None):
    function_V = function.function_space()
    if function_components is not None:
        assert isinstance(function_components, (int, str, tuple))
        assert not isinstance(function_components, list), "dolfin does not handle yet the case of a list of components"
        if isinstance(function_components, str):
            function_components = function_V.component_to_index(function_components)
        if not isinstance(function_components, tuple):
            function_V_index = (function_components, )
        else:
            function_V_index = function_components
    else:
        function_V_index = None
    if V_components is not None:
        assert isinstance(V_components, (int, str, tuple))
        assert not isinstance(V_components, list), "dolfin does not handle yet the case of a list of components"
        if isinstance(V_components, str):
            V_components = V.component_to_index(V_components)
        if not isinstance(V_components, tuple):
            V_index = (V_components, )
        else:
            V_index = V_components
    else:
        V_index = None

    V_to_function_V_mapping = dict()
    function_V_to_V_mapping = dict()

    if _function_spaces_eq(function_V, V, function_V_index, V_index):
        # Then function_V == V: do not need to extend nor restrict input function
        # Example of use case: function is the solution of an elliptic problem, V is the truth space
        if not copy:
            assert function_components is None, (
                "It is not possible to extract function components without copying the vector")
            assert V_components is None, "It is not possible to extract function components without copying the vector"
            assert weight is None, "It is not possible to weigh components without copying the vector"
            assert extended_or_restricted_function is None, (
                "It is not possible to provide an output function without copying the vector")
            return function
        else:
            if extended_or_restricted_function is None:
                output = Function(V)  # zero by default
            else:
                output = extended_or_restricted_function
                assert output.function_space() == V
            assign(_sub_from_tuple(output, V_index), _sub_from_tuple(function, function_V_index))
            if weight is not None:
                output.vector()[:] *= weight
            return output
    elif _function_spaces_lt(function_V, V, V_to_function_V_mapping, function_V_index, V_index):
        # Then function_V < V: need to extend input function
        # Example of use case: function is the solution of the supremizer problem of a Stokes problem,
        # V is the mixed (velocity, pressure) space, and you are interested in storing a extended function
        # (i.e. extended to zero on pressure DOFs) when defining basis functions for enriched velocity space
        assert copy is True, "It is not possible to extend functions without copying the vector"
        if extended_or_restricted_function is None:
            extended_function = Function(V)  # zero by default
        else:
            extended_function = extended_or_restricted_function
            assert extended_function.function_space() == V
        for (index_V_as_tuple, index_function_V_as_tuple) in V_to_function_V_mapping.items():
            assign(_sub_from_tuple(extended_function, index_V_as_tuple),
                   _sub_from_tuple(function, index_function_V_as_tuple))
        if weight is not None:
            extended_function.vector()[:] *= weight
        return extended_function
    elif _function_spaces_gt(function_V, V, function_V_to_V_mapping, function_V_index, V_index):
        # Then function_V > V: need to restrict input function
        # Example of use case: function = (y, u, p) is the solution of an elliptic optimal control problem,
        # V is the collapsed state (== adjoint) solution space, and you are
        # interested in storing snapshots of y or p components because of an aggregrated approach
        assert copy is True, "It is not possible to restrict functions without copying the vector"
        if extended_or_restricted_function is None:
            restricted_function = Function(V)  # zero by default
        else:
            restricted_function = extended_or_restricted_function
            assert restricted_function.function_space() == V
        for (index_function_V_as_tuple, index_V_as_tuple) in function_V_to_V_mapping.items():
            assign(_sub_from_tuple(restricted_function, index_V_as_tuple),
                   _sub_from_tuple(function, index_function_V_as_tuple))
        if weight is not None:
            restricted_function.vector()[:] *= weight
        return restricted_function


def _function_spaces_eq(V, W, index_V, index_W):  # V == W
    V = _sub_from_tuple(V, index_V)
    W = _sub_from_tuple(W, index_W)
    # V.sub(component) == W does not work properly
    # We thus resort to:
    assert V.ufl_domain() == W.ufl_domain()
    return V.ufl_element() == W.ufl_element()


def _function_spaces_lt(V, W, W_to_V_mapping, index_V, index_W):  # V < W
    assert V.ufl_domain() == W.ufl_domain()
    assert len(W_to_V_mapping) == 0
    V_sub_elements = _get_sub_elements(V, index_V)
    W_sub_elements = _get_sub_elements(W, index_W)
    W_sub_elements_used = dict.fromkeys(W_sub_elements.keys(), False)
    should_return_False = False
    for (index_V, element_V) in V_sub_elements.items():
        for (index_W, element_W) in W_sub_elements.items():
            if element_W == element_V and not W_sub_elements_used[index_W]:
                assert index_W not in W_to_V_mapping
                W_to_V_mapping[index_W] = index_V
                W_sub_elements_used[index_W] = True
                break
        else:  # for loop was not broken
            # There is an element in V which cannot be mapped to W, thus
            # V is larger than W
            should_return_False = True
            # Do not return immediately so that the map W_to_V_mapping
            # is filled in as best as possible

    if should_return_False:
        return False

    assert len(W_to_V_mapping) == len(V_sub_elements)  # all elements were found

    # Avoid ambiguity that may arise if there were sub elements of W that were not used but had
    # the same element type of used elements
    for (index_W_used, element_W_was_used) in W_sub_elements_used.items():
        if element_W_was_used:
            for (index_W, element_W) in W_sub_elements.items():
                if (len(index_W_used) == len(index_W)
                        and element_W == W_sub_elements[index_W_used]
                        and not W_sub_elements_used[index_W]):
                    raise RuntimeError("Ambiguity when querying _function_spaces_lt")

    return True


def _function_spaces_gt(V, W, V_to_W_mapping, index_V, index_W):  # V > W
    return _function_spaces_lt(W, V, V_to_W_mapping, index_W, index_V)


def _get_sub_elements(V, index_V):
    if index_V is not None:
        V = V.extract_sub_space(index_V)
    sub_elements = _get_sub_elements__recursive(V, index_V)
    # Re-order sub elements for increasing tuple length to help
    # avoiding ambiguities
    sub_elements__sorted_by_index_length = dict()
    for (index, element) in sub_elements.items():
        index_length = len(index)
        if index_length not in sub_elements__sorted_by_index_length:
            sub_elements__sorted_by_index_length[index_length] = OrderedDict()
        assert index not in sub_elements__sorted_by_index_length[index_length]
        sub_elements__sorted_by_index_length[index_length][index] = element
    output = OrderedDict()
    for index_length in range(min(sub_elements__sorted_by_index_length.keys()),
                              max(sub_elements__sorted_by_index_length.keys()) + 1):
        if index_length in sub_elements__sorted_by_index_length:
            output.update(sub_elements__sorted_by_index_length[index_length])
    return output


def _get_sub_elements__recursive(V, index_V):
    sub_elements = OrderedDict()
    if V.num_sub_spaces() == 0:
        if index_V is None:
            index_V = (None, )
        sub_elements[tuple(index_V)] = V.ufl_element()
        return sub_elements
    else:
        for i in range(V.num_sub_spaces()):
            index_V_comma_i = list()
            if index_V is not None:
                index_V_comma_i.extend(index_V)
            index_V_comma_i.append(i)
            sub_elements_i = _get_sub_elements__recursive(V.sub(i), index_V_comma_i)
            sub_elements.update(sub_elements_i)
        return sub_elements


def _sub_from_tuple(input_, index_as_tuple):
    if index_as_tuple is None:
        index_as_tuple = (None, )
    assert isinstance(index_as_tuple, tuple)
    assert len(index_as_tuple) > 0
    if len(index_as_tuple) == 1 and index_as_tuple[0] is None:
        return input_
    else:
        for i in index_as_tuple:
            input_ = input_.sub(i)
        return input_
