#!/usr/bin/env python
# -*- coding:gb18030 -*-
"""
File  :   layer_utils.py
Author:   zhanghao55@baidu.com
Date  :   20/12/29 19:52:01
Desc  :   
"""

import collections
import six


def is_sequence(seq):
    """
    Whether `seq` is an entry or nested structure
    """
    if isinstance(seq, dict):
        return True
    return (isinstance(seq, collections.Sequence) and
            not isinstance(seq, six.string_types))

def _sorted(dict_):
    """
    Returns a sorted list of the dict keys, with error if keys not sortable.
    """
    try:
        return sorted(six.iterkeys(dict_))
    except TypeError:
        raise TypeError("nest only supports dicts with sortable keys.")


def _yield_value(iterable):
    if isinstance(iterable, dict):
        # Iterate through dictionaries in a deterministic order by sorting the
        # keys. Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        for key in _sorted(iterable):
            yield iterable[key]
    else:
        for value in iterable:
            yield value


def _yield_flat_nest(nest):
    for n in _yield_value(nest):
        if is_sequence(n):
            for ni in _yield_flat_nest(n):
                yield ni
        else:
            yield n


def flatten(nest):
    """
    Traverse all entries in the nested structure and put them into an list.
    """
    if is_sequence(nest):
        return list(_yield_flat_nest(nest))
    else:
        return [nest]


def _sequence_like(instance, args):
    """
    Convert the sequence `args` to the same type as `instance`.
    """
    if isinstance(instance, dict):
        # Pack dictionaries in a deterministic order by sorting the keys.
        # Notice this means that we ignore the original order of `OrderedDict`
        # instances. This is intentional, to avoid potential bugs caused by mixing
        # ordered and plain dicts (e.g., flattening a dict but using a
        # corresponding `OrderedDict` to pack it back).
        result = dict(zip(_sorted(instance), args))
        return type(instance)((key, result[key])
                              for key in six.iterkeys(instance))
    elif (isinstance(instance, tuple) and hasattr(instance, "_fields") and
          isinstance(instance._fields, collections.Sequence) and
          all(isinstance(f, six.string_types) for f in instance._fields)):
        # This is a namedtuple
        return type(instance)(*args)
    else:
        # Not a namedtuple
        return type(instance)(args)


def _packed_nest_with_indices(structure, flat, index):
    """
    Helper function for pack_sequence_as.
    """
    packed = []
    for s in _yield_value(structure):
        if is_sequence(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def pack_sequence_as(structure, flat_sequence):
    """
    Pack a given flattened sequence into a given structure.
    """
    if not is_sequence(flat_sequence):
        raise TypeError("flat_sequence must be a sequence")
    if not is_sequence(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "Structure is a scalar but len(flat_sequence) == %d > 1" %
                len(flat_sequence))
        return flat_sequence[0]
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(
            "Could not pack sequence. Structure had %d elements, but flat_sequence "
            "had %d elements.  Structure: %s, flat_sequence: %s." %
            (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    _, packed = _packed_nest_with_indices(structure, flat_sequence, 0)
    return _sequence_like(structure, packed)


def map_structure(func, *structure):
    """
    Apply `func` to each entry in `structure` and return a new structure.
    """
    flat_structure = [flatten(s) for s in structure]
    entries = zip(*flat_structure)
    return pack_sequence_as(structure[0], [func(*x) for x in entries])
