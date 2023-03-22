# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class MissingSentinel:
    """One instance of this is created as MISSING_SENTINEL below.

    That instance is used to indicate that a variable lacks a value, and nothing else.

    Usually None is used for this purpose, but sometimes None is in the valid
    set of values and cannot be used to mean that a value is missing.

    This is very similar to dataclasses.MISSING, but that value has a private type."""

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING_SENTINEL = MissingSentinel()
