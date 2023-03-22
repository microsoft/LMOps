# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Optional


class CacheClient(ABC):
    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    async def get(self, args: dict) -> Optional[dict]:
        pass

    @abstractmethod
    async def upload(self, args: dict, result: dict) -> None:
        pass
