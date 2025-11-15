from typing import Mapping, Dict

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL

from deepscaler.rewards.math_utils.utils import extract_answer
import random
import re

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""


def extract_judge(text: str, invalid_judge_metric: Dict) -> int:
    invalid_judge_metric["n_judge"] += 1
    text = text.split(THOUGHT_DELIMITER_END)[-1]
    model_answer = extract_answer(text)
    if model_answer is None:
        pass
    else:
        model_answer = model_answer.strip()

    if model_answer == "Assistant 1":
        return 1
    elif model_answer == "Assistant 2":
        return 2
    else:
        # print(f"[W] Invalid model answer: {model_answer}, using randomly generated answer")
        invalid_judge_metric["n_invalid_judge"] += 1
        # return True if random.random() > 0.5 else False
        return -1


def extract_judge_for_elo(text: str, invalid_judge_metric: Mapping) -> int:
    invalid_judge_metric["n_judge"] += 1
    text = text.split(THOUGHT_DELIMITER_END)[-1]
    model_answer = extract_answer(text)
    if model_answer is None:
        pass
    else:
        model_answer = model_answer.strip()

    if model_answer == "Assistant 1":
        return 1
    elif model_answer == "Assistant 2":
        return 2
    else:
        # print(f"[W] Invalid model answer: {model_answer}, using randomly generated answer")
        invalid_judge_metric["n_invalid_judge"] += 1
        return 0
      

if __name__ == "__main__":
    text = "<think> I am omniscient. </think> To determine whether the final product being a text file indicates low effort or mediocre quality, we consider the following points:1. **Project Requirements**: If the project's requirements necessitated a text file, it might have been the chosen format.2. **Technical Constraints**: Constraints might have forced a text file, such as tool compatibility or technical limitations.3. **Complexity**: The content's complexity could justify a text file, requiring significant effort.4. **Purpose**: The project's intended format, like guides or manuals, could favor text files.5. **Collaboration**: Collaboration might have led to a text file for simplicity and ease of useFor the quality aspect:1. **Content**: Assess if the text is well-researched and organized.2. **Project Goals**: Reflect on whether the friend met key deliverables.3. **Exchange**: Discuss design decisions and challenges.4. **Supporting Materials**: Look at additional content for insight.In conclusion, the text file's format and content are more indicative of effort and quality than its mere format.Answer: \\boxed{Assistant 2 }"
    output = extract_judge(text)
    print(output)
 