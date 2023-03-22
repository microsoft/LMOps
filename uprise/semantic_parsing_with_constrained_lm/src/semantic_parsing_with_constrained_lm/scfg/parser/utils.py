# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def is_skippable(string: str):
    """A string is skippable if it's empty or begins with a '#'"""
    return not string or string[0] == "#"
