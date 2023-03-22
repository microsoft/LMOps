# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO, Tuple, Union

import jsons
import tqdm
import typer
from appdirs import user_cache_dir
from blobfile import BlobFile
from dataflow.core.dialogue import Dialogue, Turn
from dataflow.core.lispress import parse_lispress, render_compact

from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.generate import parse_and_render
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.scfg.scfg import SCFG
from semantic_parsing_with_constrained_lm.src.semantic_parsing_with_constrained_lm.domains.calflow.grammar import (
    get_templates,
    induce_grammar,
    read_function_types,
    remove_unused_types,
)

ROOT = Path(__file__).resolve().parent
GRAMMAR_DIR = ROOT / "grammar"
DATA_DIR = ROOT / "data"

CACHE_DIR = user_cache_dir("semantic_parsing_as_constrained_lm")


def dialogues_from_file(
    ids_filepath: Optional[Path],
    calflow_dialogues: List[Dialogue],
    only_dialogue_ids: bool = False,
    read_topk: Optional[int] = None,
) -> Tuple[Dict[Tuple[str, int], Turn], Dict[Tuple[str, int], Turn]]:
    """
    Given a path to a file containing pairs of dialogue ids and turn part indices, and a list of calflow dialogues,
    return a dictionary of mapping ids to matching turns and a dictionary
    mapping ids to the turns precedings those matching turns.
    If ids_filepath is not set, all dialogue turns are selected.
    If only_dialogue_ids = True, ids_filepath only contains dialogue ids and no turn ids. We select all turns in
    that case.
    """

    all_ids: Optional[Set[Union[str, Tuple[str, int]]]] = None

    if ids_filepath is not None:
        all_ids = set()
        with open(str(ids_filepath), "r") as id_file:
            for line_i, line in enumerate(id_file):
                if read_topk is not None and line_i >= read_topk:
                    break
                if only_dialogue_ids:
                    dialogue_id = line.strip()
                    all_ids.add(dialogue_id)
                else:
                    dialogue_id, turn_index_str = line.strip().split(",")
                    turn_index = int(turn_index_str)
                    all_ids.add((dialogue_id, turn_index))

    all_context_turns: Dict[Tuple[str, int], Turn] = {}
    all_turns: Dict[Tuple[str, int], Turn] = {}

    for d in calflow_dialogues:
        if all_ids is not None and only_dialogue_ids and d.dialogue_id not in all_ids:
            continue
        for index, t in enumerate(d.turns):
            if t.skip:
                continue
            # Populate all_context_turns for all turns of selected dialogues
            if index > 0:
                prev_turn = d.turns[index - 1]
                all_context_turns[(d.dialogue_id, t.turn_index)] = prev_turn
            if (
                all_ids is not None
                and not only_dialogue_ids
                and (d.dialogue_id, t.turn_index) not in all_ids
            ):
                continue
            all_turns[(d.dialogue_id, t.turn_index)] = t

    return all_turns, all_context_turns


def write_dialogues(
    scfg: SCFG,
    all_turns: Tuple[Dict[Tuple[str, int], Turn], Dict[Tuple[str, int], Turn]],
    output_filepath: Path,
) -> None:
    turns, context_turns = all_turns
    with open(f"{output_filepath}.jsonl", "w") as output_f:
        for dialogue_id, turn_index in tqdm.tqdm(turns, desc=output_filepath.name):
            t = turns[dialogue_id, turn_index]
            lispress = render_compact(parse_lispress(t.lispress))

            canonical_utterances = parse_and_render(
                scfg, " " + lispress, source_is_plan=True
            )
            try:
                # There is occasional ambiguity because of mega functions.
                # Take the shortest parse.
                canonical_utterance = min(canonical_utterances, key=len)
            except ValueError:
                canonical_utterance = None

            # The calflow grammar as written puts a space before every
            # utterance, but we don't want to include those.
            if canonical_utterance is not None and canonical_utterance[0] == " ":
                assert canonical_utterance[1] != " "
                canonical_utterance = canonical_utterance[1:]

            if turn_index > 0:
                context = context_turns[
                    (dialogue_id, turn_index)
                ].agent_utterance.original_text
            else:
                context = ""

            output_f.write(
                json.dumps(
                    {
                        "dialogueId": dialogue_id,
                        "turnIndex": t.turn_index,
                        "utterance": t.user_utterance.original_text,
                        "canonicalUtterance": canonical_utterance,
                        "lispress": lispress,
                        "context": context,
                    }
                )
            )
            output_f.write("\n")


def write_grammar(
    grammar_output_folderpath: Path,
    grammar_output_filename: str,
    templates_filepath: Path,
    types_filepaths: List[Path],
) -> None:
    """
    Given template and type information, turn them into a grammar
    and then write them to file.
    """
    templates = get_templates(templates_filepath, remove_whitespace=False)
    all_types = read_function_types(types_filepaths, templates)
    all_types = remove_unused_types(all_types)
    grammar = induce_grammar(all_types, implicit_whitespace=False)

    with open(
        str(grammar_output_folderpath / grammar_output_filename), "w"
    ) as grammar_file:
        for grammar_line in grammar:
            grammar_file.write(f"{grammar_line.rule}\n")


def dialogues_from_calflow_textio(calflow_file: TextIO) -> List[Dialogue]:
    return [jsons.loads(line.strip(), cls=Dialogue) for line in calflow_file]


@functools.lru_cache(maxsize=None)
def validation_set():
    print("Reading validation set...")
    return dialogues_from_calflow_textio(
        BlobFile(
            "https://smresearchstorage.blob.core.windows.net/semantic-parsing-with-constrained-lm/valid.dataflow_dialogues_ir.jsonl",
            streaming=False,
            cache_dir=CACHE_DIR,
        )
    )


@functools.lru_cache(maxsize=None)
def train_set():
    print("Reading train set...")
    return dialogues_from_calflow_textio(
        BlobFile(
            "https://smresearchstorage.blob.core.windows.net/semantic-parsing-with-constrained-lm/train.dataflow_dialogues_ir.jsonl",
            streaming=False,
            cache_dir=CACHE_DIR,
        )
    )


def main(dev_all: bool = typer.Option(True)):
    write_grammar(
        GRAMMAR_DIR,
        "grammar.scfg",
        DATA_DIR / "templates.csv",
        [DATA_DIR / "types.csv", DATA_DIR / "specialTypes.csv"],
    )
    print(f"Wrote grammar to {ROOT / 'grammar'}")

    scfg = SCFG.from_folder(str(GRAMMAR_DIR))

    dataset_specs = [
        ("dev_100_uniform", "ids_dev_100_uniform.txt", validation_set),
        ("test_200_uniform", "ids_dev_200_uniform.txt", validation_set),
        ("train_300_stratified", "ids_train_300_stratified.txt", train_set),
        ("train_1000_stratified", "ids_train_1000_stratified.txt", train_set),
    ]
    if dev_all:
        dataset_specs += [("dev_all", None, validation_set)]

    for output_name, ids_filename, dataset in dataset_specs:
        write_dialogues(
            scfg,
            dialogues_from_file(
                None if ids_filename is None else ROOT / "data" / ids_filename,
                dataset(),
            ),
            DATA_DIR / output_name,
        )


if __name__ == "__main__":
    typer.run(main)
