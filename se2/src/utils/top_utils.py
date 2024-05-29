# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for TOP and MTOP datasets.

References:
- TOP: (Gupta et al., 2018) Semantic Parsing for Task Oriented Dialog using
  Hierarchical Representations
- MTOP: (Li et al., 2020) MTOP: A Comprehensive Multilingual Task-Oriented
  Semantic Parsing Benchmark
"""



class TopLF:
  """Represents a logical form from TOP and MTOP datasets."""
  __slots__ = ["name", "args"]

  def __init__(self, content):
    """Initializes a TopLF object.

    Args:
      content: The content inside the bracket ([...]) of a serialized logical
        form. The first item is the intent or slot name. Each subsequent items
        is either a subtree (a TopLF object) or a leaf token (a string).
    """
    assert isinstance(content[0], str)
    self.name: str = content[0]
    self.args = content[1:]

  def serialize(self):
    """Returns the string serialization of the logical form."""
    tokens = ["[" + self.name]
    for arg in self.args:
      if isinstance(arg, TopLF):
        tokens.append(arg.serialize())
      else:
        tokens.append(arg)
    tokens.append("]")
    return " ".join(tokens)

  def __repr__(self):
    return "<TopLF {}>".format(self.serialize())


def deserialize_top(serialized):
  """Deserializes the logical form string into a TopLF object.

  Args:
    serialized: Serialized logical form string.

  Returns:
    a TopLF object if the deserialization is successful; None otherwise.
  """
  # If there is no colon after IN, the logical form needs to be deformatted.
  if serialized.startswith("[IN "):
    serialized = deformat_serialized(serialized)
  try:
    stack = [[]]
    for token in serialized.split():
      if token[0] == "[":
        # Begin a subtree: start a new stack item to collect children.
        new_stack_item = [token[1:]]
        stack.append(new_stack_item)
      elif token == "]":
        # End a subtree: create a TopLF object and add it to the parent.
        child = TopLF(stack.pop())
        stack[-1].append(child)
      else:
        # Leaf token: directly add to the parent.
        if len(stack[-1]) > 1 and isinstance(stack[-1][-1], str):
          # Combine consecutive tokens together.
          stack[-1][-1] += " " + token
        else:
          stack[-1].append(token)
    assert len(stack) == 1 and len(stack[0]) == 1
    lf = stack[0][0]
    assert isinstance(lf, TopLF)
    return lf
  except (IndexError, AssertionError):
    return None


def get_frame_top(lf):
  """Gets the frame (listing intent and slot labels) of a TopLF.

  Nested slots will be represented with a dot notation (e.g.,
  "slot1.intent2.slot3"). To avoid including all levels of nested slots,
  only slots with at least one leaf token as a direct child will be included.

  Args:
    lf: a TopLF object or a serialized logical form string.

  Returns:
    the frame string "intent-slot1-slot2-..."
  """
  if isinstance(lf, str):
    lf = deserialize_top(lf)
  args = set()

  def traverse(x, prefix="", top_level=False):
    if not top_level:
      prefix += x.name
    for v in x.args:
      if isinstance(v, TopLF):
        traverse(v, prefix + ".")
      else:
        args.add(prefix.lstrip("."))

  traverse(lf, top_level=True)
  return "-".join([lf.name] + sorted(args))


def format_serialized(serialized):
  """Formats the serialized logical form to reduce the number of tokens.

  The intent and slot labels are lower-cased and broken into words.
  Example: "[IN:SET_ALARM" --> "[IN set alarm ="

  The close brackets are also merged with the preceding token to avoid an extra
  space token after tokenization.

  Args:
    serialized: Serialized logical form string

  Returns:
    formatted logical form string.
  """
  lf_toks = []
  for tok in serialized.split():
    if tok == "]":
      lf_toks.append(tok)
    elif tok[0] == "[":
      prefix, suffix = tok[1:].split(":")
      lf_toks.append(" [" + prefix)
      lf_toks.append(" " + suffix.lower().replace("_", " "))
      lf_toks.append(" =")
    else:
      lf_toks.append(" " + tok)
  return "".join(lf_toks).strip()


def deformat_serialized(formatted):
  """Undoes the process in format_serialized."""
  lf_toks = []
  in_label = False  # Whether we are processing the intent/slot label
  for tok in formatted.replace("]", " ]").split():
    if in_label:
      if tok == "=":
        in_label = False
      else:
        lf_toks.append("_" + tok.upper())
    else:
      if tok[0] == "[":
        lf_toks.append(" " + tok + ":")
        in_label = True
      else:
        lf_toks.append(" " + tok)
  return "".join(lf_toks).replace(":_", ":").strip()