# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""General utilities

This module is designed to have minimal dependencies on the core Atmodeller package, as its
functionality is broadly applicable across different parts of the codebase. Keeping this module
lightweight also helps avoid circular imports.
"""

import logging

logger: logging.Logger = logging.getLogger(__name__)


def flatten_dictionary(d: dict, parent_key: str = "") -> dict:
    """Recursively flattens a nested dictionary, joining keys with "." to form column names.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used during recursion)

    Returns:
        Flat dictionary with dot-joined keys
    """
    items: dict = {}

    for k, v in d.items():
        new_key: str = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dictionary(v, new_key))
        else:
            items[new_key] = v

    return items


def recursively_merge_dictionaries(d1: dict, d2: dict) -> dict:
    """Recursively merges two dictionaries.

    Args:
        d1: The first dictionary
        d2: The second dictionary, which will overwrite values in the first dictionary if there are
            duplicate keys

    Returns:
        The merged dictionary
    """
    out: dict = dict(d1)

    for k, v in d2.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = recursively_merge_dictionaries(out[k], v)
            else:
                out[k] = v
        else:
            out[k] = v

    return out
