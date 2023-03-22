# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import re
from abc import ABC, abstractmethod
from typing import Pattern, Tuple

from cached_property import cached_property
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class SCFGToken(ABC):
    def render(self) -> str:
        """
        How to render this token when generating it. In most cases, you can just render the underlying value.
        Sometimes you want to modify it, like in TerminalToken.
        """
        return self.value

    @property
    @abstractmethod
    def value(self) -> str:
        """The underlying value of the token."""
        pass

    @property
    def lark_value(self) -> str:
        return self.value


class OptionableSCFGToken(SCFGToken):
    optional: bool


@dataclass(frozen=True)
class NonterminalToken(OptionableSCFGToken):
    underlying: str
    optional: bool

    @property
    def value(self):
        return self.underlying

    def is_regex(self):
        return self.underlying[0] == "/"


@dataclass(frozen=True)
class TerminalToken(OptionableSCFGToken):
    underlying: str
    optional: bool

    def render(self):
        """
        Remove the outermost quotes and unescape the rest of the quotes.
        """
        return json.loads(self.underlying)

    @property
    def value(self):
        return self.underlying

    @property
    def lark_value(self):
        return self.value + "i"


@dataclass(frozen=True)
class MacroToken(SCFGToken):
    name: str
    args: Tuple[SCFGToken, ...]

    @property
    def value(self):
        return f"{self.name}({','.join([a.value for a in self.args])})"


class EmptyToken(SCFGToken):
    @property
    def value(self):
        return ""


@dataclass(frozen=True)
class RegexToken(NonterminalToken):
    prefix: str

    def render_matching_value(self, value):
        return self.prefix + value

    @property
    def value(self):
        return self.underlying

    @property
    def lark_value(self):
        if (
            self.prefix
        ):  # We need to have this condition because lark gets mad if you give it an empty token.
            return f'"{self.prefix}" ' + self.underlying
        else:
            return self.underlying

    @cached_property
    def compiled(self) -> Pattern:
        assert self.underlying.startswith("/") and self.underlying.endswith("/")
        return re.compile(self.underlying[1:-1])


@dataclass(frozen=True)
class EntitySchema:
    name: str
    prefix: str
