#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def _infinite(scalar):  # From km3services/_tools.py
    """Creates an infinite generator from a scalar.

    Useful when zipping finite length iterators with single scalars.
    """
    while True:
        yield scalar


def _zipequalize(*iterables):  # From km3services/_tools.py
    """Creates equal length iterables for a mix of single and n-length arrays"""

    dims = set(map(len, iterables))
    if len(dims) == 1:
        return iterables

    if 1 not in dims or len(dims) > 2:
        raise ValueError("Input arrays dimensions mismatch.")

    out = []
    for it in iterables:
        if len(it) == 1:
            out.append(_infinite(it[0]))
        else:
            out.append(it)

    return out


def _pdgid2flavor(pdgid):
    """Converts PDG ID to OscProb flavor"""

    if abs(pdgid) == 12:
        return 0
    if abs(pdgid) == 14:
        return 1
    if abs(pdgid) == 16:
        return 2
    raise ValueError("Unsupported PDG ID, please use neutrinos")

