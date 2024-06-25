"""Templates for the pre-train stage"""
import dataclasses
import copy
import random

UNANSWERABLE_ANSWERS = ["Unanswerable", "No enough information", "None", "Unknown", "Not enough info", 
                        "Inconclusive", "Unresolved", "Indeterminate", "No clear answer", "Ambiguous", 
                        "Unclear", "Inexplicable", "Undetermined", "Unsolvable", 
                        "Beyond comprehension", "Inscrutable", "Inconclusive", "Undefined", "Unpredictable",
                        "Undecipherable", "Unknowable", "Beyond explanation", "Not enough information",
                        "Beyond interpretation", "Insufficient details", "No definitive response", "Not yet understood",
                        "Inexplicable", "Beyond elucidation", "Without resolution", "Unclear conclusion", "Beyond comprehension",
                        "Inconclusive evidence", "Inadequate data", "Open to interpretation", "Uncertain outcome",
                        "No clear resolution", "Beyond analysis", "Indefinite answer", "Unresolved query",
                        "Inconclusive findings", "Ambiguous conclusion", "Unresolved mystery"]

UNANSWERABLE_ANSWERS += [ans.lower() for ans in UNANSWERABLE_ANSWERS] # add lowercase() answers

UNANSWERABLE_OPTIONS = ['None of the above choices', 'None', "None of the provided choices meet the criteria",
                        "No option among those listed fulfills the requirements",
                        "There is no suitable choice among the options presented",
                        "Not one of the options provided can adequately respond",
                        "None of the selections offered provide a valid solution",
                        "No option listed is sufficient to answer the question",
                        "None of the given alternatives satisfy the conditions",
                        "There are no appropriate choices among the options given",
                        "Not a single option among those provided offers a valid response",
                        "None of the choices given align with the requirements",
                        "No option provided adequately addresses the question at hand",
                        "None of the options presented offer a viable solution",
                        "Not one of the provided options is applicable to the question",
                        "None of the choices listed can sufficiently address the question",
                        "There is no option provided that meets the necessary criteria",
                        "None of the options available are suitable for the question posed",
                        "Not a single choice among those offered meets the specified criteria",
                        "None of the alternatives provided adequately fulfill the requirements",
                        "No option among those presented adequately resolves the question",
                        "None of the available choices adequately meet the specified conditions"
                        "None of the options fit", "No option works", "Nothing fits", "No suitable choice",
                        "No viable option", "No answer among these", "None are right", "Nothing applies",
                        "No match", "No valid choice", "None of the options meet the criteria",
                        "No option satisfies the conditions", "Nothing matches the requirements",
                        "No suitable answer", "No appropriate choice", "No correct option",
                        "None of these options fit", "No answer provided", "No valid response",
                        "No matching selection"]

UNANSWERABLE_OPTIONS += [ans.lower() for ans in UNANSWERABLE_OPTIONS] # add lowercase() answers

# Few-shot patterns.
# modified from flanv2: https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py
@dataclasses.dataclass
class FewShotPattern:
  """Patterns for few-shot tasks.

  The few-shot input are composed by a few examplers followed by final_suffix:
  {exampler no. 1} + {exampler no. 2} + {exampler no. 3}... + {final_suffix}

  Each exampler has the following format:
  {example_separator} + {inputs_prefix} + {inputs} + {x_y_delimiter} + {targets_prefix} + {targets}
  """
  inputs: str
  targets: str
  inputs_prefix: str = ""
  targets_prefix: str = ""
  x_y_delimiter: str = "\n\n"
  example_separator: str = "\n\n\n"
  final_suffix: str = ""
  input_pattern: str = "{{inputs}}{final_suffix}"
  in_template_mix: bool = True

  @property
  def single_example_template(self):
    return self.example_separator + self.inputs_prefix + self.inputs + self.x_y_delimiter + self.targets_prefix + self.targets

  @property
  def single_example_template_wo_seperator(self):
    return self.inputs_prefix + self.inputs + self.x_y_delimiter + self.targets_prefix + self.targets


FEWSHOT_PATTERNS = {
    "qa": [
        # natural questions from flanv2
        FewShotPattern(
            inputs="{question}??",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer this question: {question}?",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter=" ", # "\n\n" -> " "
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter=" ",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="answer this: {question}?",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="What is the answer to this question? {question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="{answer}",
            inputs_prefix="In: ",
            targets_prefix="Out: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer this: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter=" ", # "\n\n" -> " "
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question:\n\n{question}?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        # trivia_qa
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n======\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n-----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\nWhat is the answer?",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="{answer}",
            inputs_prefix="Mei Li:\n",
            targets_prefix="Shuai Zheng:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Input question: ",
            targets_prefix="Output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question.\n\n{question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="What is the answer: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # part of math_dataset
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Solve:\n{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Teacher asked me this: {question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\nSolve this plz.",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        # part of arc (those without options)
        FewShotPattern(
            inputs="An example of a question generated based on the context?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter=" ", # "\n" -> " "
            example_separator="\n\n"),
        FewShotPattern(
            inputs="I just read this text in school today. What question was I "
            "asked after reading it?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question I was asked: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Random question?",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n", # "\n\n"-> "\n"
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a question",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write a question you would see in a school textbook based on the context.",
            targets="{question}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        # part of bigbench:simple_arithmetic_json.gen.blueridge_vocab.0_shot.30_examples
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter=" ", # "\n\n" -> " "
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Solve this: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Reply with the result:\n\n{question}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question answering problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # part of bigbench:auto_debugging.gen.blueridge_vocab.0_shot.34_examples
            FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer the following question:\n{question}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer the following:\n\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Hmm... {answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Given the question below, answer directly after the question ended:\n{question}",
            targets="{answer}",
            inputs_prefix="IN: ",
            targets_prefix="",
            x_y_delimiter=" ", # "\n\n" -> " "
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer the following question:\n{question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: so it's ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See this question:\n{question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="The quick answer is:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        # t0_question_answer
        FewShotPattern(
            inputs="{question}",
            targets="Ans: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            targets_prefix="Answer: ",
            x_y_delimiter="\n----\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer the following: {question}",
            targets="{answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Please answer this: {question}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Given the question: {question}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="The answer is:\n",
            x_y_delimiter="\n++++++++++++++++++++++++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}???",
            targets="{answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # t0_multiple_choice
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Q: {question}",
            targets="A: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="*Question*\n",
            targets_prefix="**Answer**\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            x_y_delimiter="\n",
            example_separator="\n------\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="(Question)\n",
            targets_prefix="(Answer)\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Ques: ",
            targets_prefix="Ans: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="(Q).\n",
            targets_prefix="(A).\n",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            x_y_delimiter="\n\n",
            example_separator="\n-----\n"),
        # ==== input inversion ====
        # "program_synthesis_dmcc_python_input_inversion"
        FewShotPattern(
            inputs="If this is the answer: {answer}\n What was the question?",
            targets="{question}",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="This sentence answers a question. {answer}",
            targets="{question}",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="[BEGIN]{answer}[DONE]",
            targets="{question}",
            inputs_prefix="Answer. ",
            targets_prefix="Problem. ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}\n\nProblem this solves:",
            targets="{question}",
            inputs_prefix="Solution: ",
            targets_prefix="Problem: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="If this is the answer: {answer}\nWhat's the question?",
            targets="{question}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="[BEGIN]{answer}[END]",
            targets="{question}",
            inputs_prefix="Solution: ",
            targets_prefix="Problem that it solves: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}",
            inputs_prefix="Answer: ",
            targets_prefix="Question it solves: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}",
            inputs_prefix="[solution]*",
            targets_prefix="The problem: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        # "program_synthesis_dr_repair_input_inversion"
        FewShotPattern(
            inputs="Come up with a question that leads to this answer.\n{answer}",
            targets="{question}",
            targets_prefix="Question: ",
            x_y_delimiter="\n ---- \n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="The following is a solution to a problem.\n{answer}\n\nPropose the problem.",
            targets="{question}",
            inputs_prefix="Problem: ",
            targets_prefix="HERE: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
    ],
    "qa_w_option": [
        # part of arc (those with options)
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{option_answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the question\n\n{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Choose your answer.\n\n{question}\n\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Question: ",
            targets_prefix="My Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # unified_qa_science_inst
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(inputs="{question}\n{options_}", targets="{option_answer}"),
        FewShotPattern(
            inputs="Q: {question} ---- {options_}", targets="A: {option_answer}"),
        FewShotPattern(
            inputs="{question}\n===\n{options_}\n",
            targets="{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Answer this question:\n{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Input: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{option_answer}",
            inputs_prefix="*Question*: ",
            targets_prefix="*Answer*: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="question:\n",
            targets_prefix="answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n===\n{options_}",
            targets="{option_answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="question below:\n",
            targets_prefix="answer below:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # t0_multiple_choice_separated_options
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="A: {option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter=" ", # "\n" -> " "
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}",
            inputs_prefix="input with options: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Q: {question}\n\n{options_}",
            targets="{option_answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} Now, answer this question: {question}\nA:",
            targets="{option_answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_}\nQ: {question}",
            targets="{option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{options_}\n\n{question}",
            targets="{option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
    ],
    "qa_w_cot": [
        # synth_cot_natural_questions
        FewShotPattern(
            inputs="{question}??",
            targets="Answer: {cot}. So the answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Chain-of-thought: {cot}. Answer: {answer}",
            inputs_prefix="Question: ",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question: {question}?",
            targets="{cot}. [{answer}]",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. So... {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter=" ",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}?",
            targets="logic: {cot}\n\n{answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nLet's think...",
            targets="{cot}. So {answer} is the answer.",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {cot}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n***\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Please answer this question: {question}\nGive your reasons first.",
            targets="{cot}\n\nAnswer is {answer}",
            inputs_prefix="in: ",
            targets_prefix="out: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question:\n\n{question}? Think out loud!",
            targets="{cot}\nSo, the answer is {answer}",
            inputs_prefix="Student A: ",
            targets_prefix="Student B: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this:\n\n{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        # synth_cot_drop, removed the {context} part
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Answer this question based on the article: "
            "{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="I think:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Answer this question: {question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="question: ",
            targets_prefix="Chain-of-thought: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Based on the above article, answer a question. {question}",
            targets="{cot}\nANS: {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\nExplanation: {cot}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}. Now, let me think...",
            targets="{cot}\nSo, I would say the answer to this question is {answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question: {question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} Chain-of-thought:",
            targets="{cot}. {answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # synth_cot_common_gen, {concepts} -> {question}, {target} -> {answer}
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question} Let add some explanation.",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Chain-of-thought: {cot}. The answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="input question: ",
            targets_prefix="generation process: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="",
            targets_prefix="Let's think: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} Let see...",
            targets="{cot}. The answer is {answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} Think step-by-step:",
            targets="{cot}\nThe answer is:\n\n{answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        # "cot_gsm8k", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="Answer the following question.\n{question}",
            targets="Step-by-step reasoning process: {cot}\n"
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: "),
        FewShotPattern(
            inputs="{question}",
            targets="Step-by-step reasoning process: {cot}\n"
            "So the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} Give the rationale and then the answer.",
            targets="Let's think step by step. {cot}. "
            "The answer is: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question:{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nSo the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer with step-by-step thinking: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is: {answer}",
            targets_prefix="Let's think: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nSo the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]",
            targets_prefix="[Answer]"),
        # "cot_strategyqa", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}",
            targets="My step-by-step reasoning: {cot}\n"
            "So, the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q--",
            targets_prefix="A--",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nRationale first then the answer.",
            targets="{cot}. The answer is: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Chain of thought: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="My reasoning: {cot}\nThe answer: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="*Q:* ",
            targets_prefix="*A:* ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nPlease give rationale first, then the answer.",
            targets="{cot}. The answer is: {answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {cot}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The detailed solution is: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student question: ",
            targets_prefix="Teacher response: ",
            in_template_mix=False),
        # "cot_esnli", {chian_of_thought} -> {cot}
        FewShotPattern(
            inputs="[QUESTION] {question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's solve this gradually. {cot}\n"
            "Answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="SOLUTION: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is: {answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Q] {question}",
            targets="[A] {cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think. {cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Student asked: ",
            targets_prefix="Teacher's response: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="Let's solve it slowly: "),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nExplanation: {cot}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        # "stream_aqua", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="OK... Stream of consciousness: {cot}\n"
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Stream of consciousness: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nStream of consciousness: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Answer and stream of consciousness: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. So the answer is: {answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n-----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\nStream of consciousness:{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="question:",
            targets_prefix="answer:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. So the answer is: {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Stream of consciousness: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="question: ",
            targets_prefix="answer: "),
        # "cot_creak", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\n"
            "Chain of thoughts: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="I'm thinking hard. So here's my take: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n---\n",
            inputs_prefix="Ques: ",
            targets_prefix="Ans: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\n{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think out loud. {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\n"
            "Explanation: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Ans and explanation: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Oh man, I think this is the solution: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question part\n",
            targets_prefix="Answer part\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}\n{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="**Q**\n",
            targets_prefix="**A**\n"),
        FewShotPattern(
            inputs="Question: {question}",
            targets="Let me think..... {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\n"
            "Chain of thoughts: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Answer the following question: ",
            targets_prefix="My answer and thoughts: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Here's my solution: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            inputs_prefix="[Ques]: ",
            targets_prefix="[Ans]: ",
            in_template_mix=False),
        # cot_ecqa, {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\n"
            "CoT: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ME: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think step-by-step: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's do it gradually: {cot}... "
            "So the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: "),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think step-by-step: {cot}. "
            "So the answer must be {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question}",
            targets="Let's solve it slow. {cot}... "
            "So the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="The answer is {answer}\nExplanation: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER W/ DETAILS: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let me think. {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Person A: ",
            targets_prefix="Person B: ",
            in_template_mix=False),
        # "cot_sensemaking", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Teacher: ",
            targets_prefix="Student: "),
        FewShotPattern(
            inputs="{question}",
            targets="Chain of thought: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Jax: ",
            targets_prefix="Alex: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's see... {cot}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="My step-by-step solution: {cot}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question in book:\n",
            targets_prefix="standard solution:\n"),
        FewShotPattern(
            inputs="{question}",
            targets="This should be the solution: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Jade: ",
            targets_prefix="Lux: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            inputs_prefix="Q:",
            targets_prefix="A:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="My step-by-step solution first: {cot}... "
            "The answer is [{answer}]",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[TEACHER] ",
            targets_prefix="[Student] "),
        FewShotPattern(
            inputs="{question}",
            targets="Thoughts: {cot}. "
            "The answer is [{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        # "cot_qasc", {chain-of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="So, my chain of thought: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}]",
            targets="[{cot}\n{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think first: {cot}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="(question). ",
            targets_prefix="(answer). ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}... Explanation: {cot}. "
            "That's why the answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="[{cot}]\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[[Ques]]: ",
            targets_prefix="[[Ans]]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Let's think first: {cot}... "
            "So the answer is [{answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n------\n"),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question... ",
            targets_prefix="answer... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="Here's my solution: {cot}. "
            "The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Q] ",
            targets_prefix="[A] ",
            in_template_mix=False),
        # "stream_qed", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="q... ",
            targets_prefix="a... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Quick Question: ",
            targets_prefix="My answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {cot}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student A:",
            targets_prefix="Student B:"),
        FewShotPattern(
            inputs="{question} Let's do a good job answering this.",
            targets="Stream of consciousness: {cot}... "
            "The answer is {answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nSo the answer must be {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## question\n",
            targets_prefix="## answer\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{answer}. How to explain the answer? {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="OK... {cot}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Problem:",
            targets_prefix="Solution:"),
        FewShotPattern(
            inputs="Answer this question please:\n{question}",
            targets="Stream of random thoughts: {cot}... "
            "The answer is {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}\nFINAL ANSWER: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="# QUESTION\n",
            targets_prefix="# ANSWER\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}",
            targets="{cot}. The answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        # "cot_input_inversion_gsm8k", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="Consider the Q and A. Q: {question}\nA: {answer}\nWhat is the step-by-step reasoning process?",
            targets="{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Step-by-step reasoning process: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe answer: {answer}\nWhat was the question?",
            targets="{question}",
            inputs_prefix="Reasoning and answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify this answer: {answer}",
            targets="The question is: {question}\nStep-by-step reasoning process: {cot}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}\nThe question and answer:",
            targets="{question}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="Step-by-step reasoning process: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {cot}\nThe answer: {answer}",
            targets="{question}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify [{answer}] as the answer.",
            targets="The question is: {question}\nReasoning: {cot}",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}",
            targets="The question is {question}\nThe answer is {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="[Q & A] ",
            in_template_mix=False),
        FewShotPattern(
            inputs="We have a question: {question}\nAnd an answer: {answer}\nSo how you got the answer?",
            targets="{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe answer: {answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Reverse engineering the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # "cot_input_inversion_strategyqa", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="The question is: {question}\n{cot}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}\nThe question and answer:",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="CoT and answer: ",
            targets_prefix="Do reverse engineering and find the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n{cot}",
            inputs_prefix="Known answer: ",
            targets_prefix="Now, what could be the question and solution? ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="[Q and A]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Solution: {cot}\nAnswer: {answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{answer}",
            targets="The question is: {question}\nThe rationale is: {cot}",
            inputs_prefix="The answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # "cot_input_inversion_esnli", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="Chain-of-thought: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe question and answer are below.",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="R & A: ",
            targets_prefix="Q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT] {cot}\n",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe question and answer are below.",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n*****\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="Reasoning & Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n*CoT* {cot}",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="{cot}",
            inputs_prefix="Question and answer: ",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}. So what could be the question?",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Question followed by answer: "),
        # stream_input_inversion_aqua
        FewShotPattern(
            inputs="Ques: {question}\nAns: {answer}",
            targets="CoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT] {cot}",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Reasoning: {cot}\nAns: {answer}",
            targets="Question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Ques]: {question}\n*Ans*: {answer}",
            targets="--CoT--: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n[CoT]: [{cot}]",
            inputs_prefix="Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Rationale: {cot}\nThe answer: {answer}",
            targets="Question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## Solution\n",
            targets_prefix="## What the question and answer could be\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}\nAns: {answer}",
            targets="CoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n(CoT). {cot}",
            inputs_prefix="(Ans). ",
            targets_prefix="(Question). ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        # cot_input_inversion_creak
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\n*CoT* {cot}\n",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Q&A: ",
            targets_prefix="Exp: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="Explanation and answer: ",
            targets_prefix="The corresponding question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="[{question}]\n[{answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Idea: ",
            targets_prefix="Generated [question] and [answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{answer}",
            targets="{question}\nCoT: {cot}\n",
            inputs_prefix="Ans: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\nA: {answer}",
            targets="How to explain the answer: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {cot}\nAnswer: {answer}",
            targets="What is the question? This is the question: {question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # cot_input_inversion_ecqa
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="** ",
            targets_prefix="** ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="[1] ",
            targets_prefix="[2] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="*Ans* {answer}",
            targets="*Question* {question}\n*CoT* {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="Detailed logic: ",
            targets_prefix="Question for this logic: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="CoT: ",
            targets_prefix="Q&A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="## Question and Answer ",
            targets_prefix="## Chain-of-thought ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs=">Ans< {answer}",
            targets=">Question< {question}\n>CoT< {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="Logic ==> ",
            targets_prefix="Question ==> ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question}\n{answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Q:\n",
            targets_prefix="A:\n",
            in_template_mix=False),
        # cot_input_inversion_sensemaking
        FewShotPattern(
            inputs="*Ans* {answer}",
            targets="*Question* {question}\n*CoT* {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="[{cot}]\n[{answer}]",
            targets="[{question}]",
            inputs_prefix="- ",
            targets_prefix="+ ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer -> {answer}",
            targets="Question -> {question}\nRationale -> {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="<{cot}>\n<{answer}>",
            targets="<{question}>",
            inputs_prefix="Idea: ",
            targets_prefix="Generated question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some idea for the solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: [{answer}]",
            targets="Question: [{question}]\nSolution: [{cot}]",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {question}\nAnswer: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Explanation for the following question's answer: ",
            targets_prefix=""),
        # cot_input_inversion_qasc
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic chain: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{cot}*\n*{answer}*",
            targets="*{question}*",
            inputs_prefix="Line 1: ",
            targets_prefix="Line 2: ",
            x_y_delimiter="\n",
            example_separator="\n--\n"),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Question: {question}\nCoT: {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic for the answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Solution idea: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{cot}*\n*{answer}*",
            targets="*{question}*",
            inputs_prefix="Logic of a solution: ",
            targets_prefix="The original question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Question: {question}\nCoT: {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="[[{cot}]]",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Explanation for the Ans above: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Logic for the Q&A below: ",
            targets_prefix="",
            in_template_mix=False),
        # stream_input_inversion_qed
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Ques: {question}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Stream of consciousness: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: {answer} Now, what could be the question and solution-maybe?",
            targets="Ques: {question}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n---\n"),
        FewShotPattern(
            inputs="Idea for the Q&A below: {cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{answer}",
            targets="{cot}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{answer}",
            targets="{question}",
            inputs_prefix="a: ",
            targets_prefix="q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Ans: {answer}",
            targets="Ques: {question}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n====\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\nA: {answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Some random thoughts: ",
            targets_prefix="Generated quizz based on thoughts",
            in_template_mix=False),
        
        ],
    "qa_w_option_w_cot": [
        # synth_cot_cosmos_qa, removed the {context} part
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Chain-of-thought: {cot}\nSo the answer is {option_answer}",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the following question: {question}"
            "\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Solve the following question thinking out loud: {question}\n{options_}",
            targets="{cot}. [{option_answer}]",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_}\nLet's answer this: {question}",
            targets="{cot}\nThe answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}\n{options_}\nLet's answer step by step.",
            targets="[{cot}] So the answer is {option_answer}",
            inputs_prefix="QUES: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\n{options_}",
            targets="{cot}\nThe answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}. {cot}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        # synth_cot_bool_q, remove the {context} part
        FewShotPattern(
            inputs="{question}?\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}?\n{options_}",
            targets="{cot}. So {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n==\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_}\n\n{question}?",
            targets="{cot}. Answer: {option_answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}?\n\n{options_}",
            targets="CoT: {cot}. Ans: {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}?\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n+++++++++\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="what's the best answer? {question}\n{options_}",
            targets="{cot}. Answer: {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}?\n{options_}...",
            targets="{cot}. The answer is [{option_answer}]",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question}?\n\n{options_}",
            targets="Think out loud: {cot}\nThe answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}?\n\n{options_}\nLet's think step by step.",
            targets="{cot}\nThe answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. So {option_answer}",
            inputs_prefix="input question:\n",
            targets_prefix="output answer:\n",
            x_y_delimiter="\n",
            example_separator="\n"),
        # synth_cot_paws_wiki, change the question part
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} {options_} THINK FIRST!",
            targets="{cot}. Answer: {option_answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}\nSo...",
            targets="{cot}. {option_answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}\nYour thought?",
            targets="{cot}\nThe answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="ANS: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. {option_answer}",
            inputs_prefix="IN: ",
            targets_prefix="OUT: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}\nMy answer: {option_answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="So, let's think. {cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. So [{option_answer}]",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Let's see... {cot}. So the answer is {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        # synth_cot_sentiment140, change {text} -> {question}
        FewShotPattern(
            inputs="{question} ({options_}).",
            targets="{cot}\n### {option_answer}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="My thought: {cot}. Answer: {option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{question}? {options_}",
            targets="{cot}\nAnswer: {option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="My thought: {cot}. Answer: {option_answer}.",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nSo the answer is: {option_answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Step-by-step reasoning: {cot}\nAnswer: {option_answer}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Think out loud: {question}\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\nEXPLAIN why.\n{options_}",
            targets="Answer: {option_answer}. Explanation: {cot}",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}\nWell, I think:",
            targets="{cot}\nSo the answer is: {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}\nAnalyze it:",
            targets="{cot}",
            inputs_prefix="Problem: ",
            targets_prefix="Short analysis: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # synth_cot_winogrande, change {context} -> {question}
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="sentence: ",
            targets_prefix="reasoning: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}\nLet's think step-by-step.",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="[Q]: ",
            targets_prefix="[Step-by-step]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            inputs_prefix="",
            targets_prefix="So, let's think step-by-step:",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}\nWell...",
            targets="{cot}. So the answer is {option_answer}.",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Step-by-step reasoning process: {cot}\nThe answer is {option_answer}.",
            inputs_prefix="input question: ",
            targets_prefix="output answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        # synth_cot_anli {premise} {hypothesis} -> {question}
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\n...the answer is {option_answer}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Generate a {question} with explanation",
            targets="Question: {question}\n{options_}\nExplanation: {cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Generated: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Let me think. {cot}. The answer is {option_answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Input: ",
            targets_prefix="Output: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="[Q]: ",
            targets_prefix="[A]: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}",
            inputs_prefix="",
            targets_prefix="Answer with step-by-step reasoning: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        # some typical cot dataset already incorporates the {option_} part to the {question}, 
        # here we explicitly show them in the templates to adapt to our scenario
        # "cot_gsm8k", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="Answer the following question.\n{question}\n\n{options_}",
            targets="Step-by-step reasoning process: {cot}\n"
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: "),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="Step-by-step reasoning process: {cot}\n"
            "So the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_} Give the rationale and then the answer.",
            targets="Let's think step by step. {cot}. "
            "The answer is: {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer this question:{question}\n\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}\nSo the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer with step-by-step thinking: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is: {option_answer}.",
            targets_prefix="Let's think: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}\nSo the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Question]",
            targets_prefix="[Answer]"),
        # "cot_strategyqa", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n\n{options_}",
            targets="{cot}. The answer is: {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: "),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="My step-by-step reasoning: {cot}\n"
            "So, the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q--",
            targets_prefix="A--",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}\nRationale first then the answer.",
            targets="{cot}. The answer is: {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Chain of thought: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="[Question]: ",
            targets_prefix="[Answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="My reasoning: {cot}\nThe answer: {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="*Q:* ",
            targets_prefix="*A:* ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}\nPlease give rationale first, then the answer.",
            targets="{cot}. The answer is: {option_answer}.",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}\nExplanation: {cot}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="The detailed solution is: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student question: ",
            targets_prefix="Teacher response: ",
            in_template_mix=False),
        # "cot_esnli", {chian_of_thought} -> {cot}
        FewShotPattern(
            inputs="[QUESTION] {question}\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Let's think. {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="Let's solve this gradually. {cot}\n"
            "Answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="SOLUTION: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is: {option_answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="Let's think. {cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Next Question: ",
            targets_prefix="My Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Q] {question} | {options_}",
            targets="[A] {cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="Let's think. {cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Student asked: ",
            targets_prefix="Teacher's response: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="Let's solve it slowly: "),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{option_answer}\nExplanation: {cot}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        # "stream_aqua", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="OK... Stream of consciousness: {cot}\n"
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="Stream of consciousness: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{option_answer}\nStream of consciousness: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Answer and stream of consciousness: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{cot}. So the answer is: {option_answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n-----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{option_answer}\nStream of consciousness:{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="question:",
            targets_prefix="answer:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} | {options_}",
            targets="{cot}. So the answer is: {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="OK... {cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="Stream of consciousness: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="question: ",
            targets_prefix="answer: "),
        # "cot_creak", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question} {options_}",
            targets="The answer is {option_answer}.\n"
            "Chain of thoughts: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="I'm thinking hard. So here's my take: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n---\n",
            inputs_prefix="Ques: ",
            targets_prefix="Ans: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{option_answer}\n{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="Let me think out loud. {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="The answer is {option_answer}.\n"
            "Explanation: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Ans and explanation: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="Oh man, I think this is the solution: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question part\n",
            targets_prefix="Answer part\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="{option_answer}\n{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="**Q**\n",
            targets_prefix="**A**\n"),
        FewShotPattern(
            inputs="Question: {question} {options_}",
            targets="Let me think..... {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="The answer is {option_answer}.\n"
            "Chain of thoughts: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Answer the following question: ",
            targets_prefix="My answer and thoughts: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}",
            targets="Here's my solution: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            inputs_prefix="[Ques]: ",
            targets_prefix="[Ans]: ",
            in_template_mix=False),
        # cot_ecqa, {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question} {options_}",
            targets="The answer is {option_answer}\n"
            "CoT: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ME: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let me think step-by-step: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="{cot}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let's do it gradually: {cot}... "
            "So the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: "),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let me think step-by-step: {cot}. "
            "So the answer must be {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: "),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let's solve it slow. {cot}... "
            "So the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="The answer is {option_answer}\nExplanation: {cot}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANSWER W/ DETAILS: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let me think. {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Person A: ",
            targets_prefix="Person B: ",
            in_template_mix=False),
        # "cot_sensemaking", {chain_of_thought} -> {cot}
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Teacher: ",
            targets_prefix="Student: "),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Chain of thought: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Jax: ",
            targets_prefix="Alex: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})",
            targets="Let's see... {cot}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}{options_}]",
            targets="My step-by-step solution: {cot}... "
            "So the answer is [{option_answer}]",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question in book:\n",
            targets_prefix="standard solution:\n"),
        FewShotPattern(
            inputs="{question}{options_}",
            targets="This should be the solution: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Jade: ",
            targets_prefix="Lux: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}{options_}",
            targets="{cot}\n[{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            inputs_prefix="Q:",
            targets_prefix="A:",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{question}{options_}]",
            targets="My step-by-step solution first: {cot}... "
            "The answer is [{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}{options_}",
            targets="{cot}\nThe answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[TEACHER] ",
            targets_prefix="[Student] "),
        FewShotPattern(
            inputs="{options_}\n{question}",
            targets="Thoughts: {cot}. "
            "The answer is [{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        # "cot_qasc", {chain-of_thought} -> {cot}
        FewShotPattern(
            inputs="{options_}\n{question}",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_}\n{question}",
            targets="So, my chain of thought: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="[{options_}\n{question}]",
            targets="[{cot}\n{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Ques:",
            targets_prefix="Ans:",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="Let's think first: {cot}... "
            "So the answer is [{option_answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="(question). ",
            targets_prefix="(answer). ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="{option_answer}... Explanation: {cot}. "
            "That's why the answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="q: ",
            targets_prefix="a: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="[{cot}]\n[{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[[Ques]]: ",
            targets_prefix="[[Ans]]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}",
            targets="Let's think first: {cot}... "
            "So the answer is [{option_answer}]",
            x_y_delimiter="\n--\n",
            example_separator="\n------\n"),
        FewShotPattern(
            inputs="{options_} | {question}",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="question... ",
            targets_prefix="answer... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} | {question}",
            targets="Here's my solution: {cot}. "
            "The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="[Q] ",
            targets_prefix="[A] ",
            in_template_mix=False),
        # "stream_qed", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="{options_} | {question}",
            targets="{cot}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="q... ",
            targets_prefix="a... ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} | {question}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Quick Question: ",
            targets_prefix="My answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="OK... {cot}\n{option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Student A:",
            targets_prefix="Student B:"),
        FewShotPattern(
            inputs="{question}\n{options_}\nLet's do a good job answering this.",
            targets="Stream of consciousness: {cot}... "
            "The answer is {option_answer}.",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nSo the answer must be {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## question\n",
            targets_prefix="## answer\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{option_answer}. How to explain the answer? {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="OK... {cot}\n{option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Problem:",
            targets_prefix="Solution:"),
        FewShotPattern(
            inputs="Answer this question please:\n{question}\n{options_}",
            targets="Stream of random thoughts: {cot}... "
            "The answer is {option_answer}.",
            x_y_delimiter="\n\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}\nFINAL ANSWER: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="# QUESTION\n",
            targets_prefix="# ANSWER\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}",
            targets="{cot}. The answer is {option_answer}.",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Question: ",
            targets_prefix="Answer: ",
            in_template_mix=False),
        # "cot_input_inversion_gsm8k", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="Consider the Q and A. Q: {question}\n\n{options_}\nA: {option_answer}\nWhat is the step-by-step reasoning process?",
            targets="{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Step-by-step reasoning process: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe answer: {option_answer}\nWhat was the question?",
            targets="{question}\n\n{options_}",
            inputs_prefix="Reasoning and answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify this answer: {option_answer}",
            targets="The question is: {question}\n\n{options_}\nStep-by-step reasoning process: {cot}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}\nThe question and answer:",
            targets="{question}\n\n{options_}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\n\n{options_}\nA: {option_answer}",
            targets="Step-by-step reasoning process: {cot}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {cot}\nThe answer: {option_answer}",
            targets="{question}\n\n{options_}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Come up with a question and reasoning that would justify [{option_answer}] as the answer.",
            targets="The question is: {question}\n\n{options_}\nReasoning: {cot}",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}",
            targets="The question is {question}\n\n{options_}\nThe answer is {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="[Q & A] ",
            in_template_mix=False),
        FewShotPattern(
            inputs="We have a question: {question}\n\n{options_}\nAnd an answer: {option_answer}\nSo how you got the answer?",
            targets="{cot}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe answer: {option_answer}",
            targets="{question}\n\n{options_}",
            inputs_prefix="",
            targets_prefix="Reverse engineering the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        # "cot_input_inversion_strategyqa", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question}\n\n{options_}",
            inputs_prefix="Reasoning & answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{option_answer}",
            targets="The question is: {question}\n\n{options_}\n{cot}",
            targets_prefix="Question and rationale: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}\nThe question and answer:",
            targets="{question}\n\n{options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="Question and answer: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\n{options_}\nA: {option_answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question}\n{options_}",
            inputs_prefix="CoT and answer: ",
            targets_prefix="Do reverse engineering and find the question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question}\n{options_}\n{cot}",
            inputs_prefix="Known answer: ",
            targets_prefix="Now, what could be the question and solution? ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Step-by-step reasoning process: {cot}",
            targets="{question}\n{options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="[Q and A]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\n{options_}\nA: {option_answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Solution: {cot}\nAnswer: {option_answer}",
            targets="{question}\n{options_}",
            inputs_prefix="",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{option_answer}",
            targets="The question is: {question}\n{options_}\nThe rationale is: {cot}",
            inputs_prefix="The answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # "cot_input_inversion_esnli", {chain-of-thought} -> {cot}
        FewShotPattern(
            inputs="Q: {question}\n{options_}\nA: {option_answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="Chain-of-thought: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe question and answer are below.",
            targets="{question}\n{options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question}\n{options_}",
            inputs_prefix="R & A: ",
            targets_prefix="Q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question}\n{options_}\n[CoT] {cot}\n",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question}\n{options_}\nA: {option_answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="CoT: ",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\nThe question and answer are below.",
            targets="{question} | {options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n*****\n\n",
            inputs_prefix="",
            targets_prefix=""),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} | {options_}",
            inputs_prefix="Reasoning & Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} | {options_}\n*CoT* {cot}",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question} | {options_}\nA: {option_answer}",
            targets="{cot}",
            inputs_prefix="Question and answer: ",
            targets_prefix="Explanation: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}. So what could be the question?",
            targets="{question} | {options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="",
            targets_prefix="Question followed by answer: "),
        # stream_input_inversion_aqua
        FewShotPattern(
            inputs="Ques: {question} | {options_}\nAns: {option_answer}",
            targets="CoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} | {options_}\n[CoT] {cot}",
            inputs_prefix="[Ans] ",
            targets_prefix="[Question] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Reasoning: {cot}\nAns: {option_answer}",
            targets="Question: {question} | {options_}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} | {options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="[Ques]: {question} | {options_}\n*Ans*: {option_answer}",
            targets="--CoT--: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} | {options_}\n[CoT]: [{cot}]",
            inputs_prefix="Answer: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Rationale: {cot}\nThe answer: {option_answer}",
            targets="Question: {question} | {options_}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} {options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="## Solution\n",
            targets_prefix="## What the question and answer could be\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Question: {question} {options_}\nAns: {option_answer}",
            targets="CoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n----\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} {options_}\n(CoT). {cot}",
            inputs_prefix="(Ans). ",
            targets_prefix="(Question). ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        # cot_input_inversion_creak
        FewShotPattern(
            inputs="{question} {options_}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} {options_}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} {options_}\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} {options_}\n*CoT* {cot}\n",
            inputs_prefix="*Ans* ",
            targets_prefix="*Question* ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} {options_}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Q&A: ",
            targets_prefix="Exp: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} {options_}",
            inputs_prefix="Explanation and answer: ",
            targets_prefix="The corresponding question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="[{question} {options_}]\n[{option_answer}]",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Idea: ",
            targets_prefix="Generated [question] and [answer]: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{option_answer}",
            targets="{question} {options_}\nCoT: {cot}\n",
            inputs_prefix="Ans: ",
            targets_prefix="Question: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Q: {question} {options_}\nA: {option_answer}",
            targets="How to explain the answer: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="CoT: {cot}\nAnswer: {option_answer}",
            targets="What is the question? This is the question: {question} {options_}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        # cot_input_inversion_ecqa
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} ({options_})",
            inputs_prefix="** ",
            targets_prefix="** ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} ({options_})\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})\n{option_answer}",
            targets="{cot}",
            inputs_prefix="[1] ",
            targets_prefix="[2] ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="*Ans* {option_answer}",
            targets="*Question* {question} ({options_})\n*CoT* {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} ({options_})",
            inputs_prefix="Detailed logic: ",
            targets_prefix="Question for this logic: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} ({options_})\n{option_answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="CoT: ",
            targets_prefix="Q&A: ",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question} ({options_})\n{option_answer}",
            targets="{cot}",
            inputs_prefix="## Question and Answer ",
            targets_prefix="## Chain-of-thought ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs=">Ans< {option_answer}",
            targets=">Question< {question} ({options_})\n>CoT< {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question} ({options_})",
            inputs_prefix="Logic ==> ",
            targets_prefix="Question ==> ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="{question} ({options_})\n{option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            inputs_prefix="Q:\n",
            targets_prefix="A:\n",
            in_template_mix=False),
        # cot_input_inversion_sensemaking
        FewShotPattern(
            inputs="*Ans* {option_answer}",
            targets="*Question* {question} ({options_})\n*CoT* {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n****\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {question} ({options_})\nAnswer: {option_answer}",
            x_y_delimiter="\n--\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="[{cot}]\n[{option_answer}]",
            targets="[{question} ({options_})]",
            inputs_prefix="- ",
            targets_prefix="+ ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}{options_}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer -> {option_answer}",
            targets="Question -> {question}{options_}\nRationale -> {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {question}{options_}\nAnswer: {option_answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix=""),
        FewShotPattern(
            inputs="<{cot}>\n<{option_answer}>",
            targets="<{question}{options_}>",
            inputs_prefix="Idea: ",
            targets_prefix="Generated question: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}{options_}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Question and Answer: ",
            targets_prefix="Some idea for the solution: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: [{option_answer}]",
            targets="Question: [{question}{options_}]\nSolution: [{cot}]",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Question: {options_} {question}\nAnswer: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Explanation for the following question's answer: ",
            targets_prefix=""),
        # cot_input_inversion_qasc
        FewShotPattern(
            inputs="{options_} {question}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic chain: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {options_} {question}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Chain-of-thought: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{cot}*\n*{option_answer}*",
            targets="*{options_} {question}*",
            inputs_prefix="Line 1: ",
            targets_prefix="Line 2: ",
            x_y_delimiter="\n",
            example_separator="\n--\n"),
        FewShotPattern(
            inputs="Ans: {option_answer}",
            targets="Question: {options_} {question}\nCoT: {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Logic for the answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {options_}\n{question}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Solution idea: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="*{cot}*\n*{option_answer}*",
            targets="*{options_}\n{question}*",
            inputs_prefix="Logic of a solution: ",
            targets_prefix="The original question: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Ans: {option_answer}",
            targets="Question: {options_}\n{question}\nCoT: {cot}\n",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_}\n{question}\n{option_answer}",
            targets="[[{cot}]]",
            inputs_prefix="Ques and Ans: ",
            targets_prefix="Explanation for the Ans above: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\n{options_}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n--\n",
            inputs_prefix="Logic for the Q&A below: ",
            targets_prefix="",
            in_template_mix=False),
        # stream_input_inversion_qed
        FewShotPattern(
            inputs="Ans: {option_answer}",
            targets="Ques: {question}\n{options_}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {question}\n{options_}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Stream of consciousness: ",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{question}\n{options_}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{question}\n{options_}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer: {option_answer}. Now, what could be the question and solution-maybe?",
            targets="Ques: {question}\n{options_}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n---\n"),
        FewShotPattern(
            inputs="Idea for the Q&A below: {cot}",
            targets="Q: {question}\n{options_}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="",
            targets_prefix="",
            in_template_mix=False),
        FewShotPattern(
            inputs="{options_} {question}\n{option_answer}",
            targets="{cot}",
            inputs_prefix="Ques & Ans: ",
            targets_prefix="Stream of consciousness: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="{cot}\n{option_answer}",
            targets="{options_} {question}",
            inputs_prefix="a: ",
            targets_prefix="q: ",
            x_y_delimiter="\n",
            example_separator="\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Ans: {option_answer}",
            targets="Ques: {options_} {question}\nCoT: {cot}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n====\n"),
        FewShotPattern(
            inputs="{cot}",
            targets="Q: {options_} {question}\nA: {option_answer}",
            x_y_delimiter="\n",
            example_separator="\n\n",
            inputs_prefix="Some random thoughts: ",
            targets_prefix="Generated quizz based on thoughts",
            in_template_mix=False),
    ],
    # Above are patterns for connect single QAs with each other
    # --------------------------------- Overall RC Patterns ---------------------------
    # Below are ppaterns that connects the reading context to the QAs
    "numbered_questions": [
        # "coqa", {text} -> {context}
        FewShotPattern(
            inputs="{context}\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Here are a few questions:\n{numbered_questions}\n\nAnswer the above questions after reading the text: {context}",
            targets="{numbered_answers}",
            inputs_prefix="Problem: ",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context}\n\nAnswer the following questions:"
            "\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="Q: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="Read the text and answer the questions."
            "\n{context}\n{numbered_questions}",
            targets="Numbered answers:\n{numbered_answers}",
            inputs_prefix="Question:\n",
            targets_prefix="",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n",
            in_template_mix=False),
        FewShotPattern(
            inputs="Answer the questions at the end based on the text."
            "\n\n{context}\n\n{numbered_questions}\n\nNumbered answers:",
            targets="{numbered_answers}",
            inputs_prefix="",
            targets_prefix="A: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {numbered_questions} Provide a numbered list of answers.",
            targets="{numbered_answers}",
            inputs_prefix="Question: ",
            targets_prefix="A numbered of answers: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n\n{numbered_answers}\n\nNumbered questions:",
            targets="{numbered_questions}",
            inputs_prefix="Q: ",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Make use of the article to answer the questions. {context} {numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="input: ",
            targets_prefix="numbered_answers: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="{context} {numbered_questions} Return numbered answers in your output.",
            targets="{numbered_answers}",
            inputs_prefix="input: ",
            targets_prefix="output: ",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context}\n{numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="question: ",
            targets_prefix="answer: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="{context} What are the answers to this following set of questions: {numbered_questions}",
            targets="{numbered_answers}",
            inputs_prefix="",
            targets_prefix="Answer: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
    ],
    "next_turn_dialog":[
        FewShotPattern(
            inputs="Read the dialog and predict the next turn. {dialog_}",
            targets="{answer}"),
        FewShotPattern(
            inputs="Write the response (start with \"Response:\") {dialog_}",
            targets="Response: {answer}",
            inputs_prefix="Example conversation: "),
        FewShotPattern(
            inputs="See the conversation examples, and predict the next turn. "
            "{dialog_}",
            targets="{answer}",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Complete this dialogue: {dialog_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="",
            x_y_delimiter="\n",
            example_separator="\n"),
        FewShotPattern(
            inputs="{dialog_}",
            targets="{answer}",
            inputs_prefix="",
            targets_prefix="Next turn: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="What could be the the next turn? {dialog_}",
            targets="{answer}",
            inputs_prefix="Problem: ",
            targets_prefix="A: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write another turn of this conversation. {dialog_}",
            targets="{answer}",
            inputs_prefix="QUESTION: ",
            targets_prefix="ANS: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Write another turn. {dialog_}",
            targets="{answer}",
            inputs_prefix="question: ",
            targets_prefix="turn: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See the conversation. {dialog_}",
            targets="{answer}",
            inputs_prefix="Q: ",
            targets_prefix="Next: ",
            x_y_delimiter="\n****\n",
            example_separator="\n\n\n"),
        # "predict_next_turn_dialog_input_inversion"
        FewShotPattern(
            inputs="Consider this last turn dialog: {answer}\nWhat are the preceding dialogs?",
            targets="{dialog_}",
            targets_prefix="Preceding dialog: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="Last turn conversation: {answer} The preceding conversation:",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: "),
        FewShotPattern(
            inputs="Write an example conversation that led to this. This: {answer}",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: ",
            x_y_delimiter="\n",
            example_separator="\n\n"),
        FewShotPattern(
            inputs="See the last example. Predict the preceding dialog. "
            "{answer}",
            targets="{dialog_}",
            targets_prefix="Preceding conversation: ",
            x_y_delimiter="\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="If this is the last turn dialog, what came before? {answer}",
            targets="{dialog_}",
            inputs_prefix="Problem: ",
            targets_prefix="Before this should be: ",
            x_y_delimiter="\n++++++++++\n",
            example_separator="\n\n\n"),
        FewShotPattern(
            inputs="Imagine the conversation that came before this turn\n{answer}",
            targets="{dialog_}",
            inputs_prefix="Question:\n",
            targets_prefix="Answer:\n",
            x_y_delimiter="\n\n",
            example_separator="\n\n\n"),
    ]
}

# for qa tasks that have options, we follow flan to include the templates without options too.
FEWSHOT_PATTERNS_NO_OPTIONS = copy.deepcopy(FEWSHOT_PATTERNS["qa_w_option"])
for ti, few_shot_pattern in enumerate(FEWSHOT_PATTERNS_NO_OPTIONS):
    in_template = few_shot_pattern.inputs
    out_template = few_shot_pattern.targets
    in_template = in_template.replace("{options_}\n\n", "")
    in_template = in_template.replace("\n\n{options_}", "")
    in_template = in_template.replace("\n{options_}", "")
    in_template = in_template.replace(" | {options_}", "")
    in_template = in_template.replace(" {options_}", "")
    in_template = in_template.replace(" ({options_})", "")
    in_template = in_template.replace("{options_}", "")
    in_template = in_template.replace("\n\n{options_str}", "")
    in_template = in_template.replace("\n{options_str}", "")
    
    out_template = out_template.replace("{options_}\n\n", "")
    out_template = out_template.replace("\n\n{options_}", "")
    out_template = out_template.replace("\n{options_}", "")
    out_template = out_template.replace(" | {options_}", "")
    out_template = out_template.replace(" {options_}", "")
    out_template = out_template.replace(" ({options_})", "")
    out_template = out_template.replace("{options_}", "")
    out_template = out_template.replace("\n\n{options_str}", "")
    out_template = out_template.replace("\n{options_str}", "")

    # {option_answer} -> {answer}
    in_template = in_template.replace("{option_answer}", "{answer}")
    out_template = out_template.replace("{option_answer}", "{answer}")

    few_shot_pattern.inputs = in_template
    few_shot_pattern.targets = out_template
    FEWSHOT_PATTERNS_NO_OPTIONS[ti] = few_shot_pattern

FEWSHOT_PATTERNS["qa_w_option"] += FEWSHOT_PATTERNS_NO_OPTIONS


# --------------------------- CLASSIC RC FORMAT ---------------------------------
# modified based on AdaptLLM templates: https://github.com/microsoft/LMOps/blob/main/adaptllm/utils/read.py
READ_START_STRING_CANDIDATES = [
    "Please answer some questions about the following article:\n{context}",
    "Read this article and answer questions\n{context}",
    "Answer some questions about this article:\n{context}",
    "Here are some questions about this article: {context}",
    "Article: {context}",
    "Read this text: {context}",
    "Given the article: {context}",
    "Context: {context}",
    "Use this article to answer questions: {context}",
    "Answer based on context \"{context}\"",
    "Read the following article and respond accordingly:\n\n{context}",
    "Delve into this article and answer the following questions:\n{context}",
    "Explore the text below and provide your answers\n\n{context}",
    "Engage with this article and provide your responses:\n{context}",
    "Assess your understanding of the article by answering these questions: {context}",
    "Dive into the content provided and offer your insights: {context}",
    "Your comprehension will be tested with questions based on this article\n{context}",
    "Analyze the text and formulate your responses: \"{context}\"",
    "Navigate through this article and answer the questions posed\n\n{context}",
    "Immerse yourself in this article and provide your interpretations:\n{context}",
    "Evaluate your knowledge of the article with these questions\n{context}",
    "Examine the content and demonstrate your understanding:\n{context}",
    "Interact with the article by answering the following questions:\n{context}",
    "Reflect on the text and provide your answers \"{context}\"",
    "Navigate through the provided article and answer the questions:\n{context}",
    "Interpret the article and respond to the queries below\n\n{context}",
    "Your understanding of the article will be assessed through these questions:\n{context}",
    "Engage with the material and offer your insights:\n\n{context}",
    "Demonstrate your comprehension of the article by answering these questions:\n{context}",
    "Explore the text and articulate your responses\n\n{context}",
    "Answer questions based on this article:\n{context}",
    "Read and respond:\n{context}",
    "Questions follow. Read the article and answer: {context}",
    "Use the article to answer: {context}",
    "Read and answer questions below:\n{context}",
    "Explore the article and respond:\n{context}",
    "Answer based on the following: {context}",
    "Respond to the article:\n{context}",
    "Article context\n{context}",
    "Questions to follow article:\n\n{context}",
    "Read and provide answers:\n{context}",
    "Your thoughts?\n{context}",
    "Assessment: {context}",
    "Give your take on:\n{context}",
    "Questions about this text:\n{context}",
    "Respond to the text:\n{context}",
    "Context for questions:\n{context}",
    "Discuss the article:\n{context}",
    "Provide your insights:\n{context}",
    "Answer questions here\n{context}"
]

 # include the condition where nothing is prepended to the reading material
READ_START_STRING_CANDIDATES = ['{context}'] * len(READ_START_STRING_CANDIDATES) + READ_START_STRING_CANDIDATES

READ_END_STRING_CANDIDATES = [
    " ",
    "\n",
    "\n\n",
]

COMPRE_START_STRING_CANDIDATES = [
    # connect raw text with the followed QAs
    "Please answer some questions about the above article.",
    "Answer some questions about the above article.",
    "What are the answers to these questions?",
    "Now answer these questions:",
    "Now answer the following questions",
    "What are the answers to the questions or completions:",
    "How would one answer these questions:",
    "Use evidence from the text to answer these questions:",
    "Use this above article to answer the questions:",
    "Answer the following questions based on the article",
    "Answer these questions:",
    "Based on the above article, answer questions.",
    "Write some question-answer pairs about the above article:",
    "Respond to the following questions based on the above article",
    "Upon reading the article, answer the following questions:",
    "Evaluate your understanding of the article by answering the following questions:"
    "Questions about the article above:",
    "Answer questions about the article: ",
    "Ready to solve some puzzles? Dive into the questions below based on the article.",
    "Now, let's crack these questions open!",
    "Time for a quiz! Answer the following questions:",
    "Unlock the secrets of the article with these questions:",
    "Get ready to unravel the mysteries within! Answer the questions below:",
    "Connect the dots between the article and these questions:",
    "Let the article be your guide as you tackle these questions:",
    "Engage your mind with these questions inspired by the article:",
    "Curious minds, it's your turn! Answer these questions:",
    "Article analysis time! Answer the questions below:",
    "Navigate through these questions and uncover insights from the article:",
    "Transform your understanding into answers with these questions:",
    "Explore the depths of comprehension with these questions:",
    "Challenge yourself with these questions inspired by the article:",
    "Discover new perspectives through these questions:",
    "Get inquisitive! Answer the questions below based on the article:",
    "It's quiz time! See how much you've learned by answering these questions:",
    "Solve the puzzle of the article with these questions:",
    "Put your comprehension to the test with these questions:",
    "Probe deeper into the article with these thought-provoking questions:",
    "Reflect on the article and respond to these questions:",
    "Chart your comprehension journey by answering these questions:",
    "Uncover insights from the article with these questions:",
    "Ready to explore? Answer these questions inspired by the article:",
    "Get your thinking caps on! Answer the questions below based on the article:",
    "Illuminate your understanding with these questions:",
    "Unlock the article's secrets by answering these questions:",
    "Navigate through these questions and reveal your insights from the article:",
    "Dive into the questions about the article above:",
    "Put your thinking cap on and tackle these questions about the article:",
    "Get ready for a mental workout! Here are the questions based on the article:",
    "The article has spoken, now it's your turn to answer these questions:",
    "Challenge accepted? Answer the following questions inspired by the article:",
    "Your journey through the article isn't over yet! Answer these questions:",
    "Unravel the mysteries of the article by answering these questions:",
    "Let's explore the depths of the article together. Answer the questions below:",
    "The context is your map, these questions are your compass. Answer them:",
    "Ready to decode the text's message? Start by answering these questions:",
    "Embark on a quest for knowledge! Answer the questions inspired by the passage:",
    "Connect the dots between the article and your understanding with these questions:",
    "Test your comprehension! Answer the following questions based on the context:",
    "The article has ignited your curiosity, now satisfy it by answering these questions:",
    "Challenge your intellect! Answer these questions derived from the article:",
    "The article has opened doors to new ideas. Step through them with these questions:",
    "Ready to apply what you've learned? Answer these questions about the article:",
    "Uncover hidden truths with these questions inspired by the passage:",
    "Your journey through the article isn't complete until you've answered these questions:",
    "Illuminate your understanding of the article by answering these questions:",
    "The article has sparked questions, now it's your turn to answer them:",
    "Deepen your understanding of the text snnipet with these thought-provoking questions:",
    "Navigate through the article's depths with these guiding questions:",
    "Reflect on the article's content and respond to these questions:",
    "Chart your path through the article with these insightful questions:",
    "Unearth the article's treasures by answering these questions:",
    "Ready for a challenge? Answer these questions inspired by the text:",
    "The article has laid the groundwork, now it's your turn to build upon it with these questions:",
    "Illuminate your comprehension of the article with these enlightening questions:",
    "Unlock the article's secrets by answering these revealing questions:",
    "Embark on an exploration of the article's themes with these probing questions:",
    "Craft your own set of Q&A for the above passage:",
    "Create question-answer duos inspired by the article:",
    "Exercise your knowledge by generating Q&A pairs for the article:",
    "Generate your own set of questions and answers about the above article:",
    "Put your comprehension to the test - create question-answer pairs for the context:",
    "Challenge yourself with this task: devise question-answer pairs related to the above article:",
    "Immerse yourself in active learning by formulating question-answer pairs about the article:",
    "Practice your understanding by developing question-answer pairs for the above article:",
]

# include the condition where nothing is between the read. and compre.
COMPRE_START_STRING_CANDIDATES = COMPRE_START_STRING_CANDIDATES + [''] * len(COMPRE_START_STRING_CANDIDATES)

# reading comprehension inversion
# we put the QA before the reading material
QA_START_STRING_CANDIDATES = [
    "{QA_demos}",
    "Here are a few question-answering pairs, write an article based on them:\n{QA_demos}",
    "Read these Q&A pairs and write a passage\n\n{QA_demos}",
    "Write a piece of text after reading the dialogue below\n{QA_demos}",
    "Here are some questions and their answers about an article, try to reproduce the article: {QA_demos}",
    "Exercise: {QA_demos}",
    "Read this dialogue: \"{QA_demos}\"",
    "Create an article using the following question-answer pairs:\n{QA_demos}",
    "Construct a passage using the provided Q&A examples:\n\n{QA_demos}",
    "Explore the following dialogue and craft a written piece:\n{QA_demos}",
    "Utilize the given Q&A pairs to compose a narrative:\n{QA_demos}",
    "Reproduce an article based on the questions and answers provided:\n{QA_demos}",
    "Generate content inspired by the following dialogue:\n{QA_demos}",
    "Enhance your writing skills by transforming the provided Q&A pairs into a cohesive article:\n{QA_demos}",
    "Immerse yourself in the dialogue below and produce a written response:\n{QA_demos}",
    "Practice your writing abilities by elaborating on the following Q&A examples:\n{QA_demos}",
    "Craft a narrative using the question-answer pairs provided:\n\n{QA_demos}",
    "Interpret the dialogue and create a written piece in response:\n{QA_demos}",
    "Challenge yourself to expand upon the given Q&A pairs:\n{QA_demos}",
    "Develop your storytelling skills by composing an article from the following Q&A examples:\n{QA_demos}",
    "Compose a narrative based on the questions and answers outlined below:\n{QA_demos}",
    "Engage in creative writing using the dialogue provided:\n{QA_demos}",
    "Exercise your writing muscles by transforming the Q&A pairs into an article:\n{QA_demos}",
    "Transform the dialogue into a written composition:\n{QA_demos}",
    "Harness your imagination to expand upon the following question-answer pairs:\n{QA_demos}",
    "Create an article using these Q&A examples:\n{QA_demos}",
    "Craft a passage based on these question-answer pairs:\n\n{QA_demos}",
    "Use the following dialogue to write a piece of text:\n{QA_demos}",
    "Write an article inspired by these Q&A pairs:\n{QA_demos}",
    "Reproduce an article based on these questions and answers:\n{QA_demos}",
    "Generate content from the following dialogue:\n{QA_demos}",
    "Compose an article using these Q&A pairs:\n{QA_demos}",
    "Practice your writing skills with these Q&A examples:\n{QA_demos}",
    "Craft a story using these question-answer pairs:\n\n{QA_demos}",
    "Expand upon these Q&A pairs to create an article:\n{QA_demos}",
    "Develop a narrative from these questions and answers:\n{QA_demos}",
    "Use the dialogue to inspire your writing:\n{QA_demos}",
    "Transform these Q&A pairs into an article:\n{QA_demos}",
    "Write a composition based on the dialogue:\n\n{QA_demos}",
    "Create a story using these Q&A examples:\n{QA_demos}",
    "Practice your writing skills with this dialogue:\n\n{QA_demos}",
    "Expand on these question-answer pairs to create content:\n\n{QA_demos}"
]

# # include the condition where nothing is prepended to the reading material
# QA_START_STRING_CANDIDATES = ['{QA_demos}'] * len(QA_START_STRING_CANDIDATES) + QA_START_STRING_CANDIDATES

QA_END_STRING_CANDIDATES = [
    " ",
    "\n",
    "\n\n",
    "\n\n",
    "\n\n\n",
    "\n",
    "\n\n",
    "\n\n",
]

TEXT_START_STRING_CANDIDATES = [
    # connect QAs with the text
    '{context}',
    "Please write the text.\nText: {context}",
    "Generate a passage based on the above QA pairs\nPassage: {context}",
    "Now compose a passage: {context}",
    "Write a passage based on the provided exercise.\n\nPassage:\n{context}",
    "Compose a passage using the above question-answer examples: {context}",
    "Craft a piece of text that encapsulates the QA pairs provided above.\nText: {context}",
    "Please write a text based on the question-answer pairs.\n\n{context}",
    "Generate a passage contains the knowledge info of the the given question-answer pairs.\n\nPassage:\n{context}",
    "Compose a narrative that reflects the content of the QA pairs above.\n\nNarative:\n{context}",
    "Now, produce a passage based on the problems and solutions:\n\n{context}",
    "Craft a passage inspired by the conversation.\nPassage: {context}",
    "Create a text that encompasses the essence of the QA pairs above.\n\n{context}",
    "Generate a passage based on the provided question-answer pairs.\n{context}",
    "Compose a narrative using the QA pairs as inspiration.\n\n{context}",
    "Please write a passage that reflects the content of the QA pairs.\nPassage: {context}",
    "Generate a text based on the given problems and solutions.\nText: {context}",
    "Craft a passage that captures the essence of the QA pairs above.\n\nCrafted Passage: {context}",
    "Compose a literary piece inspired by the enigmatic dance of questions and answers.\n\nLiterary piece: {context}",
    "Let your imagination soar as you craft a narrative woven from the threads of these Q&A pairs.\nOk, here is the narative:\n{context}",
    "Inscribe the whispers of knowledge into the fabric of a captivating article. Article: {context}",
    "Paint a vivid tapestry of words that echoes the symphony of questions and answers.\n\n{context}",
    "Embark on a literary voyage guided by the compass of these question-answer constellations.\n\nAnswer: {context}",
    "Breathe life into the silent dialogue of questions and answers, weaving them into a literary masterpiece.\nMasterpiece: {context}",
    "Forge a literary saga from the molten embers of curiosity and enlightenment.\nSaga: {context}",
    "Dive into the labyrinth of inquiry and emerge with a literary gem forged from the depths of understanding.\n\nGem: {context}",
    "Illuminate the shadows of uncertainty with the torchlight of your pen, crafting a narrative that resonates with wisdom.\nCrafted narrative: {context}",
    "Harness the raw energy of curiosity to sculpt a narrative that transcends the boundaries of imagination.\nNarrative: {context}",
    "Compose a narrative symphony that harmonizes the discordant notes of inquiry and insight. Symphony: {context}",
    "Fashion a literary mosaic from the fragments of knowledge scattered by the interplay of questions and answers.\n\nMosaic:\n{context}",
    "Craft a literary opus that captures the essence of the quest for knowledge, expressed through the medium of questions and answers.\nOpus:\n{context}",
    "Sculpt a narrative sculpture from the marble of understanding, chiseling away the excess to reveal the beauty within.\nSculpture:\n{context}",
    "Write a narrative that dances gracefully along the tightrope between question and answer, balancing curiosity with revelation.\nNarrative: {context}",
    "Spin a tale that spins like a whirlwind, drawing the reader into the vortex of inquiry and discovery.\nTale:\n{context}",
    "Forge a literary sword from the fire of inquiry, tempered in the waters of comprehension, ready to cut through the fog of uncertainty.\nSword:\n{context}",
    "Craft a narrative quilt stitched together from the patchwork of questions and answers, each thread contributing to the tapestry of understanding.\nQuilt:\n{context}",
    "Compose a literary kaleidoscope that refracts the spectrum of human curiosity into a dazzling array of words.\n\nKaleidoscope:\n{context}",
    "Weave a narrative spell that enchants the reader, drawing them deeper into the labyrinth of inquiry with each turn of phrase. Spell: {context}",
    "Pen a narrative sonnet that sings the praises of knowledge, each stanza a testament to the power of questions and answers. Sonnet: {context}",
    "Produce a passage that synthesizes the knowledge conveyed in the given question-answer pairs.\n\nSynthesized passage:\n{context}",
    "Craft a narrative that mirrors the essence of the conversation above.\n\nMirrored narrative:\n{context}",
    "Now, create a passage inspired by the questions and answers: {context}",
    "Compose a passage that draws inspiration from the provided QA examples.\nInspired passage: {context}",
    "Develop a text that encapsulates the core concepts discussed in the question-answer pairs above.\nEncapsulated text: {context}",
    "Generate a passage that extrapolates on the themes presented in the question-answer pairs.\nExtrapolated passage: {context}",
    "Write a narrative using the QA pairs as a foundation for inspiration.\n\n{context}",
    "Please generate a passage that captures the essence of the QA pairs.\n\nCaptured passage: {context}",
    "Develop a text based on the provided QA pairs.\n\nAnswer:\n{context}",
    "Craft a passage that embodies the key insights conveyed in the conversation above.\nEmbodied passage: {context}",
    "Now, create a passage utilizing the following QA pairs: {context}",
    "Suppose the provided question-answering pairs serve as testing tasks for a reading material. What would be the accompanying text?\nAccompanying text: {context}",
    "Imagine the conversations above function as assessments for a reading material. What would be the corresponding text?\nCorresponding text: {context}",
    "If the QA pairs provided are used as assessments for a reading material, what material would you expect to be included?\nExpected material: {context}",
    "Consider the provided dialogue as evaluation criteria for a reading material. What would be the content of the reading material?\nContent of material: {context}",
    "Picture the QA pairs above as tests for a reading material. What would be the content of the material being tested?\nTested material: {context}",
    "Imagine the provided questions and answers are tests for a reading material. What material would you anticipate students reading?\nAnticipated material: {context}",
    "Suppose the QA pairs serve as assessment prompts for a reading material. What would be the text that students need to read?\nRequired text: {context}",
    "If the QA pairs above are used to assess comprehension of a reading material, what material do you think students would need to read?\n\n{context}",
    "Consider the provided questions and answers as assessments for a reading material. What material would be assigned for students to read?\n\nAssigned material: {context}",
    "Imagine the questions and answers provided are assessments for a reading material. What text do you think students would be required to read?\n\nRequired material: {context}"
]

# # include the condition where nothing is between the QA and Text.
# TEXT_START_STRING_CANDIDATES = TEXT_START_STRING_CANDIDATES

# ---------------------------------- OPTION FORMAT -------------------------------
# All possible option start strings. We will randomly pick one to use, when
# formatting a option string.
OPT_START_STRING_CANDIDATES = [
    "",
    "OPTIONS:",
    "Possible answers:",
    "Available choices:",
    "Options:",
    "OPT:",
    "Choose from:",
    "Choose your answer from:",
    "Available options:",
    "Options are:",
    "Choices:",
    "Pick your answer from:",
    "Select from:",
    "Pick from:",
    "Select from the following.",
    "pick from the following.",
]

ROMAN_NUMS = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
    "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII",
    "XXIII", "XXIV", "XXV", "XXVI"
]
# All possible itemization strings.
# This is an array of char arrays. Each array represents 26 strings, supporting
# up to 26 option items (because there are 26 chars from A to Z). If an example
# has more than 26 options, it will error out.
# We will randomly pick one to use, when formatting a option string.
OPT_ITEM_NAME_CANDIDATES = [
    ["\n - " for _ in range(26)],
    ["\n -" for _ in range(26)],
    ["\n -- " for _ in range(26)],
    ["\n --" for _ in range(26)],
    ["\n + " for _ in range(26)],
    ["\n +" for _ in range(26)],
    ["\n * " for _ in range(26)],
    ["\n *" for _ in range(26)],
    ["\n- " for _ in range(26)],
    ["\n+ " for _ in range(26)],
    ["\n* " for _ in range(26)],
    ["\n[-] " for _ in range(26)],
    ["\n[+] " for _ in range(26)],
    [" - " for _ in range(26)],
    [" -- " for _ in range(26)],
    [" + " for _ in range(26)],
    [" * " for _ in range(26)],
    [" -" for _ in range(26)],
    [" --" for _ in range(26)],
    [" +" for _ in range(26)],
    [" *" for _ in range(26)],
    [" [+] " for _ in range(26)],
    [" [-] " for _ in range(26)],
    [" " + x.lower() + ". " for x in ROMAN_NUMS],  #  i.
    [" [" + x.lower() + "] " for x in ROMAN_NUMS],  #  [i]
    [" (" + x.lower() + ") " for x in ROMAN_NUMS],  #  (i)
    [" (" + x.lower() + "). " for x in ROMAN_NUMS],  #  (i).
    ["\n" + x.lower() + ". " for x in ROMAN_NUMS],  # \ni.
    ["\n[" + x.lower() + "] " for x in ROMAN_NUMS],  # \n[i]
    ["\n(" + x.lower() + ") " for x in ROMAN_NUMS],  # \n(i)
    ["\n(" + x.lower() + "). " for x in ROMAN_NUMS],  # \n(i).
    [" " + x.upper() + ". " for x in ROMAN_NUMS],  #  I.
    [" [" + x.upper() + "] " for x in ROMAN_NUMS],  #  [I]
    [" (" + x.upper() + ") " for x in ROMAN_NUMS],  #  (I)
    [" (" + x.upper() + "). " for x in ROMAN_NUMS],  #  (I).
    ["\n" + x.upper() + ". " for x in ROMAN_NUMS],  # \nI.
    ["\n[" + x.upper() + "] " for x in ROMAN_NUMS],  # \n[I]
    ["\n(" + x.upper() + ") " for x in ROMAN_NUMS],  # \n(I)
    ["\n(" + x.upper() + "). " for x in ROMAN_NUMS],  # \n(I).
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A].
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a].
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1].
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A).
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a).
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1).
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA).
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na).
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1).
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A).
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a).
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1).
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A).
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a).
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1).
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A).
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a).
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1).
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A).
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a).
    [" " + str(x) + "). " for x in range(1, 27)],  #  1).
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A.
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a.
    [" " + str(x) + ". " for x in range(1, 27)],  #  1.
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A]:
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a]:
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1]:
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A):
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a):
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1):
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA):
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na):
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1):
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A):
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a):
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1):
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A):
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a):
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1):
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A):
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a):
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1):
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A):
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a):
    [" " + str(x) + "). " for x in range(1, 27)],  #  1):
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A:
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a:
    [" " + str(x) + ". " for x in range(1, 27)],  #  1:
]

# All possible strings to add to the end of all items.
# Currently, we only support up to 26 items.
OPT_ITEM_END_STR_CANDIDATES = [
    ["" for _ in range(26)],
    [";" for _ in range(26)],
    # ["." for _ in range(26)],
    ["" for _ in range(26)],
]

# All possible strings to add between selected options in the multi-label question scenario
OPT_DELIMITER_CANDIDATES = [', ', '\t', '\n', '; ', ' ', ' | ']

def concat_answers(selected_options, delimiter):
    """synthesize answer string where there are multiple selected options"""
    if delimiter == ', ':
        return  ', '.join(selected_options[:-1]) + ' and ' + selected_options[-1]
    else:
        return delimiter.join(selected_options)

# modified based on: https://github.com/google-research/FLAN/blob/e9e4ec6e2701182c7a91af176f705310da541277/flan/v2/preprocessors.py#L588
def format_options(example, start_str, item_names, opt_end_str, opt_delimiter, unanswerable_option, random_seed):
    """Formats options in a variety of ways.
    args:
        start_str: start_str.
        item_names: item_name for each option.
        opt_end_str: end str for each option.

    We take array example["options"] as input, and outputs example["options_"]
    which contains a random format of the options. Note that this supports up to
    26 option items.
    For example:
    example["options"] = [["An apple", "A banana", "A cashew"]]
    example["options_"] can be:
        "Select from below.
           + Eat apple;
           + Eat banana;
           + Eat cashew;"

    Note that we change the answer string too. If the options are:
        A. Eat apple; B. Eat banana;
    Then the answer will be changed from "Eat apple"/"Eat banana" to "A."/"B.".

    Args:
        example: dictionary of inputs. It must have an "options" entry.

    Returns:
        The same example dictionary but with the additional "options_" entry.
    """
    options = copy.deepcopy(example["options"])
    random.Random(random_seed).shuffle(options) # the model tends to generate many gold options at (A) or (C), we try to mitigate this
    # Interleave item_names, options, and opt_end_str.
    options_ = [
        f"{item}{opt}{end}" for item, opt, end in zip(item_names[:len(options)], options, opt_end_str[:len(options)])
    ]

    example["options_"] = "".join([start_str] + options_)

    # Potentially changed the answer string.
    answer = example["answer"]

    # step 1, check if None of the options are correct
    if answer not in options and answer in UNANSWERABLE_OPTIONS:
        example["option_answer"] = unanswerable_option
        return example

    # step 2, check if single label question
    match_arr = [answer.strip() == opt.strip() for opt in options]
    has_match = any(match_arr)

    # Check if the item_names are unique.
    is_abc_style = len(set(item_names)) == len(item_names)

    if has_match == True:
        # single label question
        answer_maybe_abc_style = item_names[match_arr.index(True)].strip() if is_abc_style else answer
        example["option_answer"] = answer_maybe_abc_style
    else:
        raise Exception('no match answers')    

    return example


# ------------------------ Numbered Questions ---------------------------
# adapted from option item name candidates, only reserve the items having index
QUESTION_ITEM_NAME_CANDIDATES = [
    [" " + x.lower() + ". " for x in ROMAN_NUMS],  #  i.
    [" [" + x.lower() + "] " for x in ROMAN_NUMS],  #  [i]
    [" (" + x.lower() + ") " for x in ROMAN_NUMS],  #  (i)
    [" (" + x.lower() + "). " for x in ROMAN_NUMS],  #  (i).
    ["\n" + x.lower() + ". " for x in ROMAN_NUMS],  # \ni.
    ["\n[" + x.lower() + "] " for x in ROMAN_NUMS],  # \n[i]
    ["\n(" + x.lower() + ") " for x in ROMAN_NUMS],  # \n(i)
    ["\n(" + x.lower() + "). " for x in ROMAN_NUMS],  # \n(i).
    [" " + x.upper() + ". " for x in ROMAN_NUMS],  #  I.
    [" [" + x.upper() + "] " for x in ROMAN_NUMS],  #  [I]
    [" (" + x.upper() + ") " for x in ROMAN_NUMS],  #  (I)
    [" (" + x.upper() + "). " for x in ROMAN_NUMS],  #  (I).
    ["\n" + x.upper() + ". " for x in ROMAN_NUMS],  # \nI.
    ["\n[" + x.upper() + "] " for x in ROMAN_NUMS],  # \n[I]
    ["\n(" + x.upper() + ") " for x in ROMAN_NUMS],  # \n(I)
    ["\n(" + x.upper() + "). " for x in ROMAN_NUMS],  # \n(I).
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A].
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a].
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1].
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A).
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a).
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1).
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA).
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na).
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1).
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A).
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a).
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1).
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A).
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a).
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1).
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A).
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a).
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1).
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A).
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a).
    [" " + str(x) + "). " for x in range(1, 27)],  #  1).
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A.
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a.
    [" " + str(x) + ". " for x in range(1, 27)],  #  1.
    ["\n[" + chr(x) + "]. " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n[A]:
    ["\n[" + chr(x) + "]. " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n[a]:
    ["\n[" + str(x) + "]. " for x in range(1, 27)],  # \n[1]:
    ["\n(" + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n(A):
    ["\n(" + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n(a):
    ["\n(" + str(x) + "). " for x in range(1, 27)],  # \n(1):
    ["\n" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  # \nA):
    ["\n" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  # \na):
    ["\n" + str(x) + "). " for x in range(1, 27)],  # \n1):
    ["\n (" + chr(x) + "). " for x in range(ord("a"),
                                            ord("z") + 1)],  # \n (A):
    ["\n (" + chr(x) + "). " for x in range(ord("A"),
                                            ord("Z") + 1)],  # \n (a):
    ["\n (" + str(x) + "). " for x in range(1, 27)],  # \n (1):
    ["\n " + chr(x) + "). " for x in range(ord("a"),
                                           ord("z") + 1)],  # \n A):
    ["\n " + chr(x) + "). " for x in range(ord("A"),
                                           ord("Z") + 1)],  # \n a):
    ["\n " + str(x) + "). " for x in range(1, 27)],  # \n 1):
    [" (" + chr(x) + "). " for x in range(ord("a"),
                                          ord("z") + 1)],  #  (A):
    [" (" + chr(x) + "). " for x in range(ord("A"),
                                          ord("Z") + 1)],  #  (a):
    [" (" + str(x) + "). " for x in range(1, 27)],  #  (1):
    [" " + chr(x) + "). " for x in range(ord("a"),
                                         ord("z") + 1)],  #  A):
    [" " + chr(x) + "). " for x in range(ord("A"),
                                         ord("Z") + 1)],  #  a):
    [" " + str(x) + "). " for x in range(1, 27)],  #  1):
    [" " + chr(x) + ". " for x in range(ord("a"),
                                        ord("z") + 1)],  #  A:
    [" " + chr(x) + ". " for x in range(ord("A"),
                                        ord("Z") + 1)],  #  a:
    [" " + str(x) + ". " for x in range(1, 27)],  #  1:
]


# ----------------------- DIALOG FORMAT -----------------------
# All possible dialog start strings. We will randomly pick one to use in train
DIALOG_START_STRING_CANDIDATES = [
    "",
    "DIALOG:",
    "Dialog:",
    "CONVERSATION:",
    "Conversation:",
    "Convo:",
    "Read the following conversation:",
    "A 2 person conversation:",
    "A 2 person dialog:",
    "A dialog between 2 people:",
    "See the 2 person dialog:",
    "Conversation transcript:",
    "Phone call:",
    "2-way dialog:",
    "Example conversation:"
]

# All possible itemization strings for 2 person dialogs.
# We will randomly pick one to use, when formatting a dialog string.
DIALOG_ITEM_NAME_CANDIDATES = [
    ["\n - " for _ in range(1000)],
    ["\n -" for _ in range(1000)],
    ["\n -- " for _ in range(1000)],
    ["\n --" for _ in range(1000)],
    ["\n + " for _ in range(1000)],
    ["\n +" for _ in range(1000)],
    ["\n * " for _ in range(1000)],
    ["\n *" for _ in range(1000)],
    ["\n- " for _ in range(1000)],
    ["\n+ " for _ in range(1000)],
    ["\n* " for _ in range(1000)],
    ["\n[-] " for _ in range(1000)],
    ["\n[+] " for _ in range(1000)],
    [" - " for _ in range(1000)],
    [" -- " for _ in range(1000)],
    [" + " for _ in range(1000)],
    [" * " for _ in range(1000)],
    [" -" for _ in range(1000)],
    [" --" for _ in range(1000)],
    [" +" for _ in range(1000)],
    [" *" for _ in range(1000)],
    [" [+] " for _ in range(1000)],
    [" [-] " for _ in range(1000)],
    ["\n  A. " if x % 2 == 0 else "\n  B. " for x in range(1000)],  # \n  A.
    ["\nA. " if x % 2 == 0 else "\nB. " for x in range(1000)],  # \nA.
    ["\n[A]. " if x % 2 == 0 else "\n[B]. " for x in range(1000)],  # \n[A].
    ["\n[1]. " if x % 2 == 0 else "\n[2]. " for x in range(1000)],  # \n[1].
    ["\n[a]. " if x % 2 == 0 else "\n[b]. " for x in range(1000)],  # \n[a].
    ["\n(A) " if x % 2 == 0 else "\n(B) " for x in range(1000)],  # \n(A)
    ["\n(1) " if x % 2 == 0 else "\n(2) " for x in range(1000)],  # \n(1)
    ["\n[a]. " if x % 2 == 0 else "\n[b]. " for x in range(1000)],  # \n[a].
    ["\n[x]. " if x % 2 == 0 else "\n[y]. " for x in range(1000)],  # \n[x].
    ["\nSpeaker 1: " if x % 2 == 0 else "\nSpeaker 2: " for x in range(1000)],
    ["\nSpeaker A: " if x % 2 == 0 else "\nSpeaker B: " for x in range(1000)],
    ["\nPerson 1: " if x % 2 == 0 else "\nPerson 2: " for x in range(1000)],
    ["\nPerson A: " if x % 2 == 0 else "\nPerson B: " for x in range(1000)],
    ["\nSpeaker 1) " if x % 2 == 0 else "\nSpeaker 2) " for x in range(1000)],
    ["\nSpeaker A) " if x % 2 == 0 else "\nSpeaker B) " for x in range(1000)],
    ["\nPerson 1) " if x % 2 == 0 else "\nPerson 2) " for x in range(1000)],
    ["\nPerson A) " if x % 2 == 0 else "\nPerson B) " for x in range(1000)],
    [
        "\nAnonymous 1) " if x % 2 == 0 else "\nAnonymous 2) "
        for x in range(1000)
    ],
    [
        "\nAnonymous A) " if x % 2 == 0 else "\nAnonymous B) "
        for x in range(1000)
    ],
    ["\n  Person 1) " if x % 2 == 0 else "\n  Person 2) " for x in range(1000)],
    ["\n Person A) " if x % 2 == 0 else "\n Person B) " for x in range(1000)],
    [
        "\n Anonymous 1) " if x % 2 == 0 else "\n Anonymous 2) "
        for x in range(1000)
    ],
    [
        "\n  Anonymous A) " if x % 2 == 0 else "\n  Anonymous B) "
        for x in range(1000)
    ],
    ["\nPerson X: " if x % 2 == 0 else "\nPerson Y: " for x in range(1000)],
    ["\nSpeaker X) " if x % 2 == 0 else "\nSpeaker Y) " for x in range(1000)],
    ["\nP1: " if x % 2 == 0 else "\nP2: " for x in range(1000)],
    ["\n P1) " if x % 2 == 0 else "\n P2) " for x in range(1000)],
]

# All possible strings to add to the end of all items.
# Currently, we only support up to 1000 items.
DIALOG_ITEM_END_STR_CANDIDATES = [
    ["" for _ in range(1000)],
    [";" for _ in range(1000)],
    ["." for _ in range(1000)],
]

# ----------------------- INLINE_ FORMAT -----------------------
# modified based on FewShotPattern and inline_patterns of natinst_v2
@dataclasses.dataclass
class InlineFewShotPattern:
  """Patterns for inline few-shot tasks.

  The few-shot input are composed by a task context and a few examplers followed by final_suffix:
  {context} + {exampler no. 1} + {exampler no. 2} + {exampler no. 3}... + {testing example}

  Each ex_exampler has the following format:
  {ex_input} + {ex_output} + {example_separator}

  and the first_ex_exampler has no {example_separator}

  The testing example has the following format:
  {testing_input} + {testing_output}
  """
  ex_input: str
  ex_output: str
  context: str
  testing_input: str
  testing_output: str
  example_separator: str = "\n\n"
  final_suffix: str = ""
  input_pattern: str = "{{inputs}}{final_suffix}"
  in_template_mix: bool = True # NOTE: 这是啥

  @property
  def single_example_template(self):
    return self.ex_input + self.ex_output + self.example_separator

  @property
  def testing_input_output_template(self):
    return self.testing_input + self.testing_output

  @property
  def overall_template(self):
    return self.context + '{ex_example_str}' + '{test_example}'

INLINE_FEWSHOT_PATTERNS = {
    # the explanation, if exists, should be placed after the final_answer
    'explain_afterwards': [
    InlineFewShotPattern(
        context="You will be given a context of a task first, then some examples. "
         "Follow the examples to solve a new instance of the task.\n{context}\n\n",
        ex_input="{ex_input}",
        ex_output="\nSolution: {ex_output}\nWhy? {ex_explanation}",
        testing_input="New input: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Given the task background, example input & output, solve the new "
         "input case.\n{context}\n",
        ex_input="Example: {ex_input}",
        ex_output="\nOutput: {ex_output}\n{ex_explanation}",
        testing_input="New input case for you: {input}",
        testing_output="\nOutput: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Teacher: {context}\nTeacher: Now, understand the task background? If "
         "you are still confused, see the following example(s):\n",
        ex_input="{ex_input}",
        ex_output="\nSolution: {ex_output}\nReason: {ex_explanation}",
        testing_input="Now, solve this instance: {input}",
        testing_output="\nStudent: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n\n",
        ex_input="Example input: {ex_input}",
        ex_output="\nExample output: {ex_output}\nExample explanation: {ex_explanation}",
        testing_input="Now, solve this instance: {input}",
        testing_output="\nStudent: {output}",
        example_separator="\n"),
    InlineFewShotPattern(
        context="Context: {context}\nSee tasks below:\n",
        ex_input="Problem: {ex_input}",
        ex_output="\nSolution: {ex_output}\nExplanation: {ex_explanation}",
        testing_input="Problem: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n",
        ex_input="QA Example: {ex_input}",
        ex_output="\nExample solution: {ex_output}\nExample explanation: {ex_explanation}",
        testing_input="Problem: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n",
        ex_input="One task example: {ex_input}",
        ex_output="\nSolution is here: {ex_output}\nExplanation: {ex_explanation}",
        testing_input="Now, solve this: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Part 1. Context\n{context}\nPart 2. Task Example\n",
        ex_input="{ex_input}",
        ex_output="\nAnswer: {ex_output}\nExplanation: {ex_explanation}",
        testing_input="Part 3. Exercise\n{input}",
        testing_output="\nAnswer: {output}",
        example_separator="\n"),
    InlineFewShotPattern(
        context="{context}\n\nLet me give you some task examples: ",
        ex_input="{ex_input}",
        ex_output="\nThe answer to this example can be: {ex_output}\nHere is why: {ex_explanation}",
        testing_input="OK. solve this:\n{input}",
        testing_output="\nAnswer: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\nQ&A Examples are below.\n",
        ex_input="Q: {ex_input}",
        ex_output="\nA: {ex_output}\nRationale: {ex_explanation}",
        testing_input="Q: {input}",
        testing_output="\nA: {output}",
        example_separator="\n"),
    ],
    # no explicit explanation after the output part
    'no_explicit_explain': [
    InlineFewShotPattern(
        context="You will be given a context of a task first, then some examples. "
         "Follow the examples to solve a new instance of the task.\n{context}\n\n",
        ex_input="{ex_input}",
        ex_output="\nSolution: {ex_output}",
        testing_input="New input: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Given the task background, example input & output, solve the new "
         "input case.\n{context}\n",
        ex_input="Example: {ex_input}",
        ex_output="\nOutput: {ex_output}",
        testing_input="New input case for you: {input}",
        testing_output="\nOutput: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Teacher: {context}\nTeacher: Now, understand the task background? If "
         "you are still confused, see the following example(s):\n",
        ex_input="{ex_input}",
        ex_output="\nSolution: {ex_output}",
        testing_input="Now, solve this instance: {input}",
        testing_output="\nStudent: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n\n",
        ex_input="Example input: {ex_input}",
        ex_output="\nExample output: {ex_output}",
        testing_input="Now, solve this instance: {input}",
        testing_output="\nStudent: {output}",
        example_separator="\n"),
    InlineFewShotPattern(
        context="Context: {context}\nSee task examples below:\n",
        ex_input="Problem: {ex_input}",
        ex_output="\nSolution: {ex_output}",
        testing_input="Problem: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n",
        ex_input="Example: {ex_input}",
        ex_output="\nExample solution: {ex_output}",
        testing_input="Problem: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\n",
        ex_input="One example: {ex_input}",
        ex_output="\nSolution is here: {ex_output}",
        testing_input="Now, solve this: {input}",
        testing_output="\nSolution: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="Part 1. Context\n{context}\nPart 2. Task Example\n",
        ex_input="{ex_input}",
        ex_output="\nAnswer: {ex_output}",
        testing_input="Part 3. Exercise\n{input}",
        testing_output="\nAnswer: {output}",
        example_separator="\n"),
    InlineFewShotPattern(
        context="{context}\n\nLet me give you some QA examples: ",
        ex_input="{ex_input}",
        ex_output="\nThe answer to this example can be: {ex_output}",
        testing_input="OK. solve this:\n{input}",
        testing_output="\nAnswer: {output}",
        example_separator="\n\n"),
    InlineFewShotPattern(
        context="{context}\nQ&A Examples are below.\n",
        ex_input="Q: {ex_input}",
        ex_output="\nA: {ex_output}",
        testing_input="Q: {input}",
        testing_output="\nA: {output}",
        example_separator="\n"),
    ]
}

# --------------------------------- Format Read Compre --------------------------------
def parse_QA_list(QA_list, opt_start, opt_item_name, opt_item_end, opt_delimiter, unanswerable_option, random_seed):
    """Parse each QA pair in the QA_list to qa / qa_w_option / qa_w_cot / qa_w_cot_w_option"""
    new_QA_list = []
    for idx, qa_entry in enumerate(QA_list):
        try:
            # step 1. extract cot if exists
            if "\nLet's think step by step." in qa_entry['Q'] or "\nTherefore, the answer is " in qa_entry['A']:
                try:
                    assert qa_entry['Q'].endswith("\nLet's think step by step."), f"qa_entry['Q']: {qa_entry['Q']} does NOT ends with `\nLet's think step by step.`"
                    question_wo_cot = qa_entry['Q'][:-len("\nLet's think step by step.")].strip()
                    assert len(question_wo_cot) > 0, f'question_wo_cot: `{question_wo_cot}` is empty'

                    if len(qa_entry['A'].split("\nTherefore, the answer is ")) == 2:
                        cot, final_answer =  qa_entry['A'].split("\nTherefore, the answer is ")
                        cot, final_answer = cot.strip(), final_answer.strip()
                        if cot.endswith('.'): cot = cot[:-1] # remove the . at the end, we would add them in the fs template
                        assert len(cot) > 0 and len(final_answer) > 0, f'invalid cot: {cot} or invalid final_answer: {final_answer}'
                    else:
                        raise Exception(f"fail to extract cot and final answer from qa_entry['A']: {qa_entry['A']}")

                    has_cot = True

                except Exception as e:
                    continue
            else:
                question_wo_cot = qa_entry['Q']
                final_answer = qa_entry['A']
                has_cot = False

            # step 2. extract option if exists
            if '\nOptions:\n- ' in question_wo_cot:
                # parse question_str to raw question and the option_list
                assert len(question_wo_cot.split('\nOptions:\n- ')) == 2, f"invalid option question: {question_wo_cot}"
                question_wo_opt_wo_cot, option_str = question_wo_cot.split('\nOptions:\n- ')
                assert len(question_wo_opt_wo_cot) > 0, f'invalid question_wo_opt_wo_cot: {question_wo_opt_wo_cot}'
                option_list = option_str.split('\n- ')
                assert len(option_list) > 0, f'Failed to parse option list from: {option_str}'
                # format option_str and answer
                kw_dic = {'question': question_wo_opt_wo_cot, 'options': option_list, 'answer': final_answer}
                kw_dic = format_options(example=kw_dic, start_str=opt_start, item_names=opt_item_name, opt_end_str=opt_item_end, opt_delimiter=opt_delimiter, unanswerable_option=unanswerable_option, random_seed = random_seed*idx)
                                                                                                                                # random seed is used to shuffle the options
                                                                                                            # we hope each time we can use a different random seed for diversity
                if has_cot:
                    kw_dic['qa_mode'] = 'fs_option_cot_pattern'
                    kw_dic['cot'] = cot
                else:
                    kw_dic['qa_mode'] = 'fs_option_pattern'
            else:
                kw_dic = {'question': question_wo_cot, 'answer': final_answer}
                if has_cot:
                    kw_dic['qa_mode'] = 'fs_basic_cot_pattern'
                    kw_dic['cot'] = cot
                else:
                    kw_dic['qa_mode'] = 'fs_basic_pattern'
        except Exception as e:
            continue
        new_QA_list.append(kw_dic)
    return new_QA_list


def get_patterns(pattern_dict, random_seed, min_num_of_QAs):
    """sample the collection of patterns to update pattern_dict"""

    def _random_choose(candidates, weights=None):
        if weights is None:
            return random.Random(random_seed).choice(candidates)
        else:
            # choose based on probabilities
            return random.Random(random_seed).choices(candidates, weights=weights, k=1)[0]

    """Explaination for rc_mode
    Classic: the classic reading comprehension mode where the reading material is followed by a series of QA pairs, each is like `Question: xxx\nAnswer: xxx`
    Dialog: the reading material is followed by a two-person conversation like `Person1: xxx, Person2: xxx`
    Numbered_questions: in the followed comprehension test, we first present numerbed_questions, and ask the model to answer them all after all the questions
    Inline_FS: we first show a few examples as reference and then ask the model to answer a testing question
    Next_turn_dialog: first present one turn of dialog and then ask the model to generate more one turn"""

    rc_modes = ['Classic', 'Dialog']
    rc_weights = [95, 5] # set the probs of non_classic modes low
    if min_num_of_QAs >= 2: 
        rc_modes += ['Numbered_Questions', 'Inline_FS', 'Next_turn_dialog']
        rc_weights = [80, 5, 5, 5, 5]

    pattern_dict['rc_mode'] = _random_choose(rc_modes, weights = rc_weights)

    # rc_inversion = True means we first present context after the QAs
    pattern_dict['rc_inversion'] = _random_choose([True, False], weights = [5, 95]) # the reverse mode is very strange now, set as a very low ratio

    # rc_inversion
    pattern_dict['qa_start'] = _random_choose(QA_START_STRING_CANDIDATES)
    pattern_dict['qa_end'] = _random_choose(QA_END_STRING_CANDIDATES)
    pattern_dict['text_start'] = _random_choose(TEXT_START_STRING_CANDIDATES)

    if pattern_dict['rc_mode'] == 'Classic':
        pattern_dict['read_start'] = _random_choose(READ_START_STRING_CANDIDATES)
        pattern_dict['read_end'] = _random_choose(READ_END_STRING_CANDIDATES)
        pattern_dict['compre_start'] = _random_choose(COMPRE_START_STRING_CANDIDATES)
    elif pattern_dict['rc_mode'] in ['Dialog', 'Next_turn_dialog']:
        pattern_dict['dia_start'] = _random_choose(['\n\n', '\n', ' ', ' | ']) + _random_choose(DIALOG_START_STRING_CANDIDATES)

        # here we reverse the order of DIALOG_ITEM_NAME_CANDIDATES 
        # because we emperically found with the original order, we often sample the same dia_item_name with opt_itm_name
        dia_item_names = _random_choose(DIALOG_ITEM_NAME_CANDIDATES[::-1]) 
        
        dia_inputs_prefix = dia_item_names[0]
        dia_targets_prefix = dia_item_names[1]
        dia_end = _random_choose(DIALOG_ITEM_END_STR_CANDIDATES)[0]
        
        # the diag pattern already has some delimiter between each turn, here we add more
        dia_example_separator = _random_choose(['','\n'])
        
        # we just replace some args of fs_patterns to fit for the dialog pattern
        dia_fs_basic_pattern = copy.deepcopy(pattern_dict['fs_basic_pattern'])
        dia_fs_option_pattern = copy.deepcopy(pattern_dict['fs_option_pattern'])
        dia_fs_basic_cot_pattern = copy.deepcopy(pattern_dict['fs_basic_cot_pattern'])
        dia_fs_option_cot_pattern = copy.deepcopy(pattern_dict['fs_option_cot_pattern'])

        for pattern in [dia_fs_basic_pattern, dia_fs_option_pattern, dia_fs_basic_cot_pattern, dia_fs_option_cot_pattern]:
            pattern.inputs_prefix = dia_inputs_prefix
            pattern.targets_prefix = dia_targets_prefix
            pattern.x_y_delimiter = dia_end
            pattern.example_separator = dia_example_separator
        
        pattern_dict['fs_basic_pattern'] = dia_fs_basic_pattern
        pattern_dict['fs_option_pattern'] = dia_fs_option_pattern
        pattern_dict['fs_basic_cot_pattern'] = dia_fs_basic_cot_pattern
        pattern_dict['fs_option_cot_pattern'] = dia_fs_option_cot_pattern

        pattern_dict['next_turn_dialog'] = _random_choose(FEWSHOT_PATTERNS['next_turn_dialog'])

    elif pattern_dict['rc_mode'] == 'Numbered_Questions':
        # we re-use the OPT_ITEM_END_STR_CANDIDATES for sampling the qa_item_name
        # reverse the order to avoid sampling the same qa_item_name with the opt_item_name
        pattern_dict['qa_item_name'] = _random_choose(QUESTION_ITEM_NAME_CANDIDATES)
        pattern_dict['qa_item_end'] = _random_choose(OPT_ITEM_END_STR_CANDIDATES[::-1])
        pattern_dict['numbered_questions'] = _random_choose(FEWSHOT_PATTERNS['numbered_questions'])
    elif pattern_dict['rc_mode'] == 'Inline_FS':
        pattern_dict['inline_fs_pattern_yes_explain'] = _random_choose(INLINE_FEWSHOT_PATTERNS['explain_afterwards'])
        pattern_dict['inline_fs_pattern_no_explain'] = _random_choose(INLINE_FEWSHOT_PATTERNS['no_explicit_explain'])
    return pattern_dict


def format_one_pt_rc(read_entry, pattern_dict):
    if len(read_entry['QA_list']) == 0:
        return read_entry['context']

    if pattern_dict['rc_mode'] in ['Classic', 'Dialog']:
        qa_demo_list = []
        for qa_kw_dic in read_entry['QA_list']:
            fs_pattern = pattern_dict[qa_kw_dic['qa_mode']] 
            one_qa_demo = fs_pattern.single_example_template_wo_seperator.format(**qa_kw_dic)
            qa_demo_list.append(one_qa_demo)
        example_separator = fs_pattern.example_separator
        qa_demos = example_separator.join(qa_demo_list)

        if pattern_dict['rc_inversion'] == False:
            if pattern_dict['rc_mode'] == 'Classic':
                one_pt_rc = pattern_dict['read_start'].replace('{context}', read_entry['context'])
                if pattern_dict['compre_start'] == '': 
                    # when pattern_dict['compre_start'] = '', 
                    # we assign pattern_dict['read_end'] = '', or the line gap between read and compre would be very large
                    pattern_dict['read_end'] = ''
                one_pt_rc += pattern_dict['read_end'] + pattern_dict['compre_start']
            elif pattern_dict['rc_mode'] == 'Dialog':
                one_pt_rc = read_entry['context'] + pattern_dict['dia_start']
            one_pt_rc += example_separator + qa_demos
        elif pattern_dict['rc_inversion'] == True:
            one_pt_rc = pattern_dict['qa_start'].replace('{QA_demos}', qa_demos) + pattern_dict['qa_end'] + pattern_dict['text_start'].replace('{context}', read_entry['context'])
        else:
            raise Exception

    elif pattern_dict['rc_mode'] == 'Next_turn_dialog':
        # dialog history
        dialog_ = ''
        for qa_kw_dic in read_entry['QA_list'][:-1]:
            fs_pattern = pattern_dict[qa_kw_dic['qa_mode']] 
            one_qa_demo = fs_pattern.single_example_template.format(**qa_kw_dic)
            dialog_ += one_qa_demo
       # dialog_ = dialog_.lstrip() # remove the starting \n or space

        # next_turn to be predicted
        next_turn_dic = read_entry['QA_list'][-1]
        next_turn_pattern = pattern_dict[next_turn_dic['qa_mode']] 
        answer = next_turn_pattern.single_example_template_wo_seperator.format(**next_turn_dic)

        # format the whole rc
        rc_template = pattern_dict['next_turn_dialog'].single_example_template_wo_seperator
        one_pt_rc = read_entry['context'] + pattern_dict['next_turn_dialog'].example_separator + rc_template.format(**{'dialog_': dialog_, 'answer': answer})
    elif pattern_dict['rc_mode'] == 'Numbered_Questions':
        # format numebred_quetsions / answers
        numbered_q_list = []
        numbered_a_list = []
        for qa_kw_dic in read_entry['QA_list']:
            fs_pattern = pattern_dict[qa_kw_dic['qa_mode']]
            numbered_q_list.append(fs_pattern.inputs.format(**qa_kw_dic))
            numbered_a_list.append(fs_pattern.targets.format(**qa_kw_dic))
        
        item_names = pattern_dict['qa_item_name'][:len(numbered_q_list)]
        item_ends = pattern_dict['qa_item_end'][:len(numbered_q_list)]
        numbered_questions = "".join([f"{item}{question}{end}" for item, question, end in zip(item_names, numbered_q_list, item_ends)])
        numbered_answers = "".join([f"{item}{answer}{end}" for item, answer, end in zip(item_names, numbered_a_list, item_ends)])

        # format the whole rc
        rc_template = pattern_dict['numbered_questions'].single_example_template_wo_seperator
        one_pt_rc = rc_template.format(**{'context': read_entry['context'], 'numbered_questions': numbered_questions, 'numbered_answers': numbered_answers})
    elif pattern_dict['rc_mode'] == 'Inline_FS':
        if all('cot' in qa_kw_dic for qa_kw_dic in read_entry['QA_list']):
            inline_fs_pattern = pattern_dict['inline_fs_pattern_yes_explain']
        else:
            inline_fs_pattern = pattern_dict['inline_fs_pattern_no_explain']

        # format ex_examples
        ex_example_str = ''
        for qa_kw_dic in read_entry['QA_list'][:-1]:
            # inline_fs_pattern explicitly shows the cot part after the output part, 
            # so here we do NOT use the cot_pattern
            # 'fs_option_cot_pattern' - > 'fs_option_pattern', 'fs_basic_cot_pattern' -> 'fs_basic_pattern'
            qa_mode = qa_kw_dic['qa_mode'].replace('_cot', '')
            fs_pattern = pattern_dict[qa_mode]
            ex_dic = {}
            ex_dic['ex_input'] = fs_pattern.inputs.format(**qa_kw_dic)
            ex_dic['ex_output'] = fs_pattern.targets.format(**qa_kw_dic)

            if 'cot' in qa_kw_dic: 
                ex_dic['ex_explanation'] = qa_kw_dic['cot']

            ex_example = inline_fs_pattern.single_example_template.format(**ex_dic)
            ex_example_str += ex_example
        
        # format testing example
        test_qa_kw_dic = read_entry['QA_list'][-1]
        test_qa_mode = test_qa_kw_dic['qa_mode'].replace('_cot', '')
        test_fs_pattern = pattern_dict[test_qa_mode]
        test_dic = {}
        test_dic['input'] = test_fs_pattern.inputs.format(**test_qa_kw_dic)
        test_dic['output'] = test_fs_pattern.targets.format(**test_qa_kw_dic)

        test_example = inline_fs_pattern.testing_input_output_template.format(**test_dic)
        
        # format the whole rc
        rc_template = inline_fs_pattern.overall_template
        one_pt_rc = rc_template.format(**{'context': read_entry['context'], 'ex_example_str': ex_example_str, 'test_example': test_example})
    return one_pt_rc