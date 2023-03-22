# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import dataclasses
import re
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Union


@dataclass(frozen=True)
class Placeholder:
    name: str


@dataclass(frozen=True)
class Template:
    placeholders: Set[str]
    template: List[Union[str, Placeholder]]
    whitespace_removed: bool

    arg_pattern: ClassVar[re.Pattern] = re.compile(r"\$([^ :]*)")

    def is_terminal(self) -> bool:
        return len(self.placeholders) == 0

    def is_empty(self) -> bool:
        return len(self.placeholders) == 0 and not any(self.template)

    def render(self, replacements: Dict[str, str]) -> str:
        assert self.placeholders == replacements.keys()
        if self.whitespace_removed:
            return " ".join(
                replacements[x.name] if isinstance(x, Placeholder) else f'"{x}"'
                for x in self.template
                if x
            )
        else:
            return " ".join(
                replacements[x.name] if isinstance(x, Placeholder) else f'!"{x}"'
                for x in self.template
                if x
            )

    @staticmethod
    def from_string(s: str, remove_whitespace: bool = True) -> "Template":
        """
        Given a template string s like "$event before $dateTime", construct a Template where
        where each token in the string is identified as a string literal or a placeholder like
        Template.template = [Placeholder(event), "before", Placeholder(dateTime)]
        """
        template: List[Union[str, Placeholder]] = []
        for i, x in enumerate(re.split(Template.arg_pattern, s)):
            if i % 2 == 1:
                template.append(Placeholder(x))
            elif remove_whitespace:
                template.append(x.strip())
            else:
                template.append(x)
        placeholders = {t.name for t in template if isinstance(t, Placeholder)}
        return Template(placeholders, template, remove_whitespace)


EMPTY_TEMPLATE = Template(placeholders=set(), template=[], whitespace_removed=False)


@dataclass
class Templates:
    getter_templates: Dict[Tuple[str, str], Optional[Template]] = dataclasses.field(
        default_factory=dict
    )
    setter_function_templates: Dict[str, List[Template]] = dataclasses.field(
        default_factory=dict
    )


@dataclass(frozen=True)
class Arg:
    name: str
    type: str


@dataclass(frozen=True)
class Typed(ABC):
    pass


@dataclass(frozen=True)
class Getter(Typed):
    type: str
    field_name: str
    template: Optional[Template]


@dataclass(frozen=True)
class Enum(Typed):
    type: str


@dataclass(frozen=True)
class Function(Typed):
    name: str
    args: List[Arg]
    templates: List[Template]
    named_args: bool


@dataclass(frozen=True)
class Setter(Typed):
    type: str
    args: Tuple[Arg, ...]
    type_template: str
    arg_templates: List[Template]


@dataclass(frozen=True)
class GrammarLine:
    nonterminal: str
    rule: str
    nonterminals_used: Set[str]


def extract_setter_field_and_type(s: str) -> Tuple[str, str]:
    """
    Setters are formatted like: Setter:type.field
        e.g. Setter:Constraint[Event].subject
    """
    suffix = s.split(":")[1].split(".")
    return_type = suffix[0]
    field = suffix[1]

    return return_type, field


def extract_getter_field_and_type(s: str) -> Tuple[str, str]:
    """
    Getters are formatted like: type.field
        Date.day
    """
    suffix = s.split(".")
    return_type = suffix[0]
    field = suffix[1]

    return return_type, field


def get_templates_from_line(
    split_line: List[str], remove_whitespace: bool = True
) -> List[Template]:
    """
    Get the templates for a function from a split string.
    Function templates are formatted like:
        function_name,template1,template2,...
    Setter templates are formatted like:
        type,type_template,arg_1_template,arg_2_template,...
        Note that type_template can be empty.
    Getter templates are formatted like:
        type.field,template

    The string can have any number of trailing commas

    """
    template_strings = split_line[1:]

    # Get rid of the trailing commas. We can't do this using a filter because there may be
    # empty templates in the middle e.g. Constraint[Event],,called $subject,,,,
    i = next(
        (i for i, template in reversed(list(enumerate(template_strings))) if template),
        -1,
    )

    all_templates = [
        Template.from_string(template, remove_whitespace)
        for template in template_strings[: i + 1]
    ]
    return all_templates


def get_templates(
    templates_filename: Path, remove_whitespace: bool = True
) -> Templates:

    templates = Templates()

    with open(str(templates_filename), "r") as functions_file:
        all_lines = csv.reader(functions_file, escapechar="\\")
        for line in all_lines:
            function_name = line[0]

            if is_getter(function_name):
                function_name, field = function_name.split(".")
                getter_templates = get_templates_from_line(line, remove_whitespace)
                templates.getter_templates[function_name, field] = (
                    getter_templates[0] if getter_templates else None
                )
            else:
                templates.setter_function_templates[
                    function_name
                ] = get_templates_from_line(line, remove_whitespace)

    return templates


def is_getter(function_name: str) -> bool:
    return (
        "." in function_name
        and not function_name.endswith("apply")
        and not function_name.endswith("construct")
        and not function_name.endswith("?")
        and not function_name == "List.Nil"
    )


def read_function_types(
    filenames: List[Path], templates: Templates
) -> Dict[str, List[Typed]]:
    """
    Read type system. There are Functions, Getters, Setters, and Enums.
    Functions are formatted like:
        function_name,return_type,arg_1: arg_1_type,...,arg_n: arg_n_type
    Getters are formatted like:
        type.field,return_type,arg_name: type
    Setters are formatted like:
        Setter:type.field,type,type:type,arg_name: arg_type
        The same type can be present multiple times.
    Enums are formatter like:
        enum_name,return_type

    Returns a dictionary mapping return types to a list of Typed that can produce that type.
    """
    types: Dict[str, List[Typed]] = defaultdict(list)
    setters: Dict[str, Set[Arg]] = defaultdict(set)
    for filename in filenames:
        with open(str(filename), "r") as csvfile:
            csvreader = csv.reader(csvfile)
            for function_name, return_type, *rest in csvreader:

                if function_name.startswith("Setter:"):
                    _, field = extract_setter_field_and_type(function_name)
                    _, field_arg = rest
                    argument_name, argument_type = field_arg.split(":")
                    setters[return_type].add(Arg(argument_name, argument_type.strip()))
                elif is_getter(function_name):
                    object_type, field = extract_getter_field_and_type(function_name)
                    types[return_type].append(
                        Getter(
                            object_type,
                            field,
                            templates.getter_templates.get((object_type, field), None),
                        )
                    )
                elif function_name == "item":
                    # Special getters that don't know their type
                    object_arg, *_ = rest
                    object_type = object_arg.strip().split(":")[1].strip()
                    types[return_type].append(Getter(object_type, "item", None))
                elif (
                    return_type.startswith("Lambda")
                    and function_name != "Lambda.construct"
                ):
                    types[return_type].append(Enum(function_name))
                elif function_name[0] == "#":
                    types[return_type].append(Enum(function_name[1:]))
                else:

                    args = [
                        Arg(a[0], a[1].strip())
                        for a in [arg.strip().split(":") for arg in rest if arg]
                    ]

                    takes_named_args = function_name[0].isupper()
                    types[return_type].append(
                        Function(
                            function_name,
                            args,
                            templates.setter_function_templates.get(function_name, []),
                            named_args=takes_named_args,
                        )
                    )

        for return_type in setters:
            type_template = (  # The type template is the first template in the list of templates.
                templates.setter_function_templates[return_type][0].template[0]
                if return_type in templates.setter_function_templates
                else ""
            )
            assert isinstance(type_template, str)
            types[return_type].append(
                Setter(
                    return_type,
                    tuple(setters[return_type]),
                    type_template,
                    templates.setter_function_templates[return_type][1:]
                    if return_type in templates.setter_function_templates
                    else [],
                )
            )

    return types


def remove_unused_types(types: Dict[str, List[Typed]]) -> Dict[str, List[Typed]]:
    """
    If a function uses a type which does not appear in our type system, remove that function.
    Note that removing functions may case types to no longer appear, so we must continue to
    call this until no more changes are made.

    Note that this mutates the original types dictionary.
    """
    not_converged = True
    while not_converged:
        not_converged = False
        for t in list(types.keys()):
            for f in list(types[t]):
                # For every function, if one of the arg types does not appear in our type system,
                # remove the function.
                if isinstance(f, Function):
                    for arg in f.args:
                        if arg.type not in types:
                            types[t].remove(f)
                            not_converged = True
                        if len(types[t]) == 0:
                            del types[t]

                # For every setter, remove all the args whose types don't appear in our type system.
                if isinstance(f, Setter):
                    new_args = tuple([arg for arg in f.args if arg.type in types])
                    if len(new_args) != len(f.args):
                        not_converged = True
                        types[t].remove(f)
                        types[t].append(
                            Setter(f.type, new_args, f.type_template, f.arg_templates)
                        )

                if isinstance(f, Getter):
                    if f.type not in types:
                        not_converged = True
                        types[t].remove(f)
                    if len(types[t]) == 0:
                        del types[t]

    return types


def transform_into_nonterminal(s: str) -> str:
    new_nonterminal = s.strip().lower().replace("[", "_").replace("]", "_")
    return new_nonterminal


def induce_grammar(
    types: Dict[str, List[Typed]], implicit_whitespace: bool = True
) -> List[GrammarLine]:
    grammar = []
    for t, fs in types.items():  # pylint: disable=too-many-nested-blocks
        return_type = transform_into_nonterminal(t)
        for f in fs:
            if isinstance(f, Function):
                arg_name_to_nonterminal_names = {
                    arg.name: transform_into_nonterminal(arg.type) for arg in f.args
                }
                if f.templates:
                    expected_args = set(arg.name for arg in f.args)
                    template = next(
                        template
                        for template in f.templates
                        if expected_args == template.placeholders
                    )
                    arg_string_utterance = (
                        template.render(arg_name_to_nonterminal_names)
                        if not template.is_empty()
                        else "#e"
                    )
                elif implicit_whitespace:
                    arg_string_utterance_middle = ' "and" '.join(
                        f'"{arg_name} set to" {nonterminal}'
                        for arg_name, nonterminal in arg_name_to_nonterminal_names.items()
                    )
                    if arg_string_utterance_middle:
                        arg_string_utterance_middle = (
                            ' "with" ' + arg_string_utterance_middle
                        )
                    arg_string_utterance = (
                        f'"(call {f.name}" {arg_string_utterance_middle} ")"'
                    )
                else:
                    arg_string_utterance_middle = ' !" and " '.join(
                        f'!"{arg_name} set to " {nonterminal}'
                        for arg_name, nonterminal in arg_name_to_nonterminal_names.items()
                    )
                    if arg_string_utterance_middle:
                        arg_string_utterance_middle = (
                            ' !" with " ' + arg_string_utterance_middle
                        )
                    arg_string_utterance = (
                        f'!"(call {f.name}" {arg_string_utterance_middle} !")"'
                    )

                if f.named_args:
                    arg_string_plan = "  ".join(
                        f'" :{arg_name}" {nonterminal}'
                        for arg_name, nonterminal in arg_name_to_nonterminal_names.items()
                    )
                else:
                    arg_string_plan = "  ".join(
                        f"{nonterminal}"
                        for _, nonterminal in arg_name_to_nonterminal_names.items()
                    )

                grammar.append(
                    GrammarLine(
                        return_type,
                        f'{return_type} -> {arg_string_utterance} empty , " ({f.name}" {arg_string_plan} ")" empty',
                        set(arg_name_to_nonterminal_names.values()),
                    )
                )
            elif isinstance(f, Getter):
                nonterminal = transform_into_nonterminal(f.type)
                if f.template:
                    arg_string_utterance = f.template.render({f.type: nonterminal})
                    arg_string_utterance = arg_string_utterance.replace('""', "")

                    grammar.append(
                        GrammarLine(
                            return_type,
                            f'{return_type} -> {arg_string_utterance} empty , " (:{f.field_name}" {nonterminal} ")" empty',
                            {nonterminal},
                        )
                    )
                elif implicit_whitespace:
                    grammar.append(
                        GrammarLine(
                            return_type,
                            f'{return_type} -> "{f.field_name} of" {nonterminal} empty , " (:{f.field_name}" {nonterminal} ")" empty',
                            {nonterminal},
                        )
                    )
                else:
                    grammar.append(
                        GrammarLine(
                            return_type,
                            f'{return_type} -> !"{f.field_name} of " {nonterminal} empty , " (:{f.field_name}" {nonterminal} ")" empty',
                            {nonterminal},
                        )
                    )

            elif isinstance(f, Setter):
                f_type = transform_into_nonterminal(f.type)
                all_args_nonterminal = f"{return_type}_{f_type}_args"
                for arg in f.args:
                    arg_nonterminal = transform_into_nonterminal(arg.type)
                    template = next(
                        (
                            arg_t
                            for arg_t in f.arg_templates
                            if arg_t.placeholders == {arg.name}
                        ),
                        None,
                    )

                    if template:
                        utterance = template.render({arg.name: arg_nonterminal})
                        if implicit_whitespace:
                            grammar_line_args_utt = (
                                f"{utterance} {all_args_nonterminal}"
                            )
                        else:
                            # Normally, expansions don't begin with a space when
                            # implicit_whitespace is False.  However, this is an
                            # exception because we don't want to put a trailing
                            # space before each appearance of `all_args_nomterinal`
                            # as we may have zero arguments.
                            #
                            # Example:
                            #  constraint_event_ -> !"event" constraint_event__constraint_event__args
                            #  constraint_event__constraint_event__args -> constraint_list_attendee__ constraint_event__constraint_event__args
                            #  constraint_event__constraint_event__args -> !" " !"for " constraint_duration_ constraint_event__constraint_event__args
                            #
                            # With the above rules, we can parse all of:
                            # - "event",
                            # - "event with Ben" (using Constraint[List[Attendee]])
                            # - "event for 30 minutes" (using Constraint[Duration])"
                            # - "event for 30 minutes with Ben"
                            # - "event with Ben for 30 minutes"
                            # and so on.
                            if f.type_template:
                                grammar_line_args_utt = (
                                    f'!" " {utterance} {all_args_nonterminal}'
                                )
                            else:
                                grammar_line_args_utt = (
                                    f"{utterance} {all_args_nonterminal}"
                                )
                    elif implicit_whitespace:
                        # NB: The next line should start with "with" rather than
                        # " with" since the lack of ! means there is already a
                        # built-in space. However we're leaving it unchanged so
                        # that we can reproduce the old grammar exactly.
                        grammar_line_args_utt = f'" with {arg.name} set to" {arg_nonterminal}  {all_args_nonterminal}'
                    else:
                        grammar_line_args_utt = f'!" with {arg.name} set to " {arg_nonterminal}  {all_args_nonterminal}'

                    grammar_line_args_plan = (
                        f'" :{arg.name}" {arg_nonterminal}  {all_args_nonterminal}'
                    )
                    grammar.append(
                        GrammarLine(
                            all_args_nonterminal,
                            f"{all_args_nonterminal} -> {grammar_line_args_utt} empty , {grammar_line_args_plan} empty ",
                            {arg_nonterminal, all_args_nonterminal},
                        )
                    )

                if f.type_template:
                    if implicit_whitespace:
                        type_template_str = f'"{f.type_template}"'
                    else:
                        type_template_str = f'!"{f.type_template}"'
                else:
                    type_template_str = ""

                grammar.append(
                    GrammarLine(
                        return_type,
                        f'{return_type} -> {type_template_str} {all_args_nonterminal} empty , " ({f.type}" {all_args_nonterminal} ")" empty',
                        {all_args_nonterminal},
                    )
                )
                grammar.append(
                    GrammarLine(
                        all_args_nonterminal, f"{all_args_nonterminal} -> #e, #e", set()
                    )
                )

    grammar.append(GrammarLine("empty", "empty -> #e , #e", set()))

    return grammar
