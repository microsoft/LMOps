# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os  # pylint: disable=unused-import
from typing import Union

# This can be used to annotate arguments that are supposed to be file paths.
StrPath = Union[str, "os.PathLike[str]"]
