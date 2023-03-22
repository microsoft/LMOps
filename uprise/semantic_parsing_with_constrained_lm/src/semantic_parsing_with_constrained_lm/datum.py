# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Optional, TypeVar


@dataclass(frozen=True, eq=True)
class Datum:
    dialogue_id: Optional[str]
    turn_part_index: Optional[int]
    agent_context: Optional[str]
    natural: str


@dataclass(frozen=True, eq=True)
class FullDatum(Datum):
    canonical: str


FullDatumSub = TypeVar("FullDatumSub", bound=FullDatum, contravariant=True)
DatumSub = TypeVar("DatumSub", bound=Datum, contravariant=True)
