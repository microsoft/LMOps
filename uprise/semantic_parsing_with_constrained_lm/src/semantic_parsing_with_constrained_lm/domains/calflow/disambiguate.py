# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re

from dataflow.core.lispress import Lispress, parse_lispress

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.util.util import flatten


def no_alwaystrue_under_getintrasalient(f: Lispress):
    """
    GetIntraSalient should always be called with an AlwaysTrueConstraint
    """
    if isinstance(f, str) or len(f) == 0:
        return False
    if f[0] == "getIntraSalient":
        return "AlwaysTrueConstraint" not in f[1][0]
    return any(no_alwaystrue_under_getintrasalient(x) for x in f)


def weird_do(f: Lispress):
    """
    The arguments to Do should be Yield expressions.
    """
    if isinstance(f, str) or len(f) == 0:
        return False
    if f[0] == "do":
        if len(f) < 2:
            return True
        return not f[1][0] == "Yield"
    return any(weird_do(x) for x in f)


def weird_yield(f: Lispress):
    """Expressions called by Yield should begin with an alphabetic character."""
    return f[0] == "Yield" and not f[2][0][0].isalpha()


def is_banned(lispress_str: str) -> bool:
    """
    Check if the lispress_str contains any banned patterns.
    """
    try:
        form = parse_lispress(lispress_str)
    except AssertionError:
        return True
    has_double_yield = "(Yield :output (Yield" in lispress_str
    preflight_no_create = lispress_str.count(
        "CreatePreflightEventWrapper"
    ) - lispress_str.count(
        "(CreateCommitEventWrapper :event (CreatePreflightEventWrapper"
    )
    no_yield = (
        "Yield" not in lispress_str
        and "Fence" not in lispress_str
        and "Pleasantry" not in lispress_str
        and "UserPauseResponse" not in lispress_str
        and "RepeatAgent" not in lispress_str
        and "DoNotConfirm" not in lispress_str
    )
    place_multi_results_and_weather = (
        "FindPlaceMultiResults" in lispress_str and "Weather" in lispress_str
    )
    place_not_weather = (
        "FindPlace" in lispress_str
        and not "FindPlaceMultiResults" in lispress_str
        and "Weather" not in lispress_str
    )
    return (
        has_double_yield
        or no_yield
        or weird_do(form)
        or place_multi_results_and_weather
        or place_not_weather
        or no_alwaystrue_under_getintrasalient(form)
        or preflight_no_create > 0
    )


def has_findperson_under_create(f: Lispress) -> bool:
    """CreateAddPerson needs to be used in create/clobber situations"""
    if isinstance(f, str) or len(f) == 0:
        return False
    if f[0] == "CreatePreflightEventWrapper":
        return "FindAddPerson" in flatten(f)
    if f[:4] == [
        "ClobberWrapper",
        ":oldLocation",
        ["Constraint[Constraint[Event]]"],
        ":new",
    ]:
        return "FindAddPerson" in flatten(f[4:])
    return any(has_findperson_under_create(x) for x in f)


def has_noncanonical_andconstraint(f: Lispress) -> bool:
    """andConstraints should be left-branching"""
    if isinstance(f, str) or len(f) == 0:
        return False
    if f[0] == "andConstraint":
        if len(f) < 3:
            return True
        for child in f[2:]:
            if (
                isinstance(child, list)
                and len(child) > 0
                and child[0] == "andConstraint"
            ):
                # andConstraint can only be the first child
                return True
    return any(has_noncanonical_andconstraint(x) for x in f)


def score_auto_grammar_plan(lispress_str: str) -> float:
    """
    Assigns a rule-based score to a Lispress str, with higher scores being preferred.
    We use this to disambiguate when a canonical utterance has multiple possible parses.
    """

    mega_functions = [
        "FindNumNextEventWrapper",
        "PathAndTypeConstraint",
        "ResponseWrapper",
        "ClobberWrapper",
        "NewClobberWrapper",
        "ChooseCreateEventWrapper",
        "ChooseUpdateEventWrapper",
        "ChooseCreateEventFromConstraintWrapper",
        "ChooseUpdateEventFromConstraintWrapper",
        "ChoosePersonFromConstraintWrapper",
        "CreateAddPerson",
        "FindAddPerson",
        "UpdateWrapper",
        "DeleteWrapper",
        "NumberOrdinal",
    ]

    if is_banned(lispress_str):
        return -float("inf")

    form = parse_lispress(lispress_str)

    # dispreferred
    num_chars = len(lispress_str)
    num_parens = lispress_str.count("(")
    always_true = lispress_str.count("AlwaysTrueConstraint")  # prefer `Constraint`
    yield_execute = lispress_str.count("(Yield :output (Execute")
    bare_preflight_wrapper = lispress_str.count(":output (UpdatePreflightEventWrapper")
    bad_output_equals = lispress_str.count(":output (?=")
    bad_output_id = lispress_str.count(":output (:id")
    bad_output_yield = lispress_str.count(":output (Yield")

    # preferred
    singletons = lispress_str.count(
        "(singleton (:results (FindEventWrapperWithDefaults"
    )
    # sometimes bad
    singleton_as_arg_to_describe = lispress_str.count(
        ":output (singleton (:results (FindEventWrapperWithDefaults"
    )
    event_on_date_output = (
        lispress_str.count(":output (EventOn")
        + lispress_str.count(":output (EventAfter")
        + lispress_str.count(":output (EventBefore")
        + lispress_str.count(":output (EventDuring")
    )

    # For creates and updates, we use `do the Recipient "John"` (AttendeeListHasRecipient), but for queries
    # we need `any Recipient "John" (FindAddPerson)
    attendee_create_or_update = (
        lispress_str.count("CreateCommitEventWrapper")
        + lispress_str.count("UpdateWrapper")
    ) * lispress_str.count("FindAddPerson")
    num_mega_functions = sum(
        lispress_str.count(mega_function) for mega_function in mega_functions
    )

    top_level_without_as_person = lispress_str.count(":output (CreateAddPerson")
    top_level_person_name_like = lispress_str.count(":output (PersonWithNameLike")
    event_date_time_separate = lispress_str.count("(EventAtTime :event (Event")

    empty_constraint_match = re.search(
        "(?<!extensionConstraint )\\(Constraint\\[([A-Za-z]+)\\]\\)", lispress_str
    )
    bad_empty_constraint = (
        1
        if empty_constraint_match
        and empty_constraint_match.group(1) != "Event"
        and empty_constraint_match.group(1) != "Recipient"
        else 0
    )

    bad_recip = lispress_str.count(
        "RecipientWithNameLike :constraint (Constraint[Recipient])"
    )

    bad_year = (
        1 if re.search(":year #\\(Number [0-9]{1,3}[\\.\\)]", lispress_str) else 0
    )

    return (
        -1000.0 * has_findperson_under_create(form)
        + -1000.0 * has_noncanonical_andconstraint(form)
        + -30.0 * always_true
        + -30.0 * attendee_create_or_update
        + -100.0
        * (
            bare_preflight_wrapper
            + bad_output_equals
            + bad_output_id
            + bad_output_yield
            + bad_recip
        )
        + -1000.0 * bad_year
        + -50.0 * singleton_as_arg_to_describe
        + -30.0 * (event_date_time_separate + event_on_date_output)
        + -1000.0 * bad_empty_constraint
        + -20.0 * yield_execute
        + -100.0 * (top_level_without_as_person + top_level_person_name_like)
        + -20.0 * weird_yield(form)
        + -5.0 * num_parens
        + -0.1 * num_chars
        + 30.0 * singletons
        + 20.0 * num_mega_functions
    )
