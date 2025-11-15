"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from deepscaler.system_prompts import ORM_PROMPT
from deepscaler.utils import call_gemini_llm, call_oai_rm_llm

import re

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.JUDGE, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        model_answer = extract_answer(model_solution)

        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        match = re.search(r"(Solution|Assistant) (\d+)", model_answer)
        if match:
            model_answer = match.group(2)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def skywork_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.JUDGE, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem= "I see that my friend has finished their project. However, their final product is still a text file. Does this necessarily mean that they didn't put in much effort or that their project is mediocre? Choose the better answer from the following two responses: Solution 1: The final product being a text file has no direct correlation with the effort or quality of the project. Here are a few reasons why: 1.  **Project Requirements**: The project requirements might have specifically asked for a text file as the final output. For example, if the project was about generating a report or creating a script, a text file might be the most appropriate format. 2.  **Technical Constraints**: The project might have had technical constraints that made a text file the most feasible option. For instance, if the project involved working with a specific tool or platform that only supports text files, this could be the reason. 3.  **Complexity**: Just because the final product is a text file doesn't mean it's simple. The content within the text file could be highly complex, requiring a lot of effort and expertise to create. 4.  **Purpose**: The purpose of the project could be such that a text file is the most suitable format. For example, if the project was about creating a guide or a manual, a text file might be more practical than a more visually-oriented format. 5.  **Collaboration**: The project might have involved collaboration with others, and a text file could have been chosen for its simplicity and ease of collaboration. Solution 2: Having a final product in the form of a text file does not inherently indicate a lack of effort or quality. Here are some reasons why a text file might be a valid final product: 1. **Intended format**: Depending on the project's goals, a text file might be the most suitable format. For example, if the project involved data analysis, a text file could be a convenient way to present the results, especially if the data is intended to be easily machine-readable. 2. **Interim or prototyping stage**: Your friend might have chosen to deliver a text file as a proof-of-concept or an interim result, with the intention of polishing the presentation later. This could be a deliberate decision to focus on the underlying work rather than the presentation. 3. **Technical requirements**: Certain projects, such as those involving scriptwriting, configuration files, or code, naturally result in text files. In these cases, a text file is a sensible and expected outcome. 4. **Purposely minimalist**: Your friend might have deliberately opted for a simple text file to emphasize the content over flashy presentation. This approach can be effective for conveying complex information in a straightforward manner. 5. **Time constraints or priorities**: Your friend might have had to prioritize the project's core aspects over the presentation, due to time constraints or other factors. In this case, a text file could be a pragmatic solution to get the project completed on time. To assess the quality and effort put into the project, consider the following: 1. **Evaluate the content**: Take a closer look at the content within the text file. Is it well-researched, organized, and clearly written? 2. **Understand the project goals**: Review the project's objectives and requirements. Did your friend meet the key deliverables and achieve the desired outcomes? 3. **Ask your friend**: Have an open conversation with your friend about their project. Ask about their design decisions, challenges, and what they learned during the process. This will give you insight into their thought process and effort. 4. **Look for supporting materials**: Check if your friend has created any supplementary materials, such as diagrams, flowcharts, or presentations, that demonstrate their work and thought process. Avoid making assumptions based solely on the format of the final product. Instead, focus on the content, project goals, and your friend's explanations to gauge the quality and effort invested in the project. Let's analyze this step by step and decide which solution is better, and then answer \\boxed{Solution 1} or \\boxed{Solution 2}.", problem_type=RewardType.JUDGE, model_response="<think> I am omniscient. </think> To determine whether the final product being a text file indicates low effort or mediocre quality, we consider the following points:1. **Project Requirements**: If the project's requirements necessitated a text file, it might have been the chosen format.2. **Technical Constraints**: Constraints might have forced a text file, such as tool compatibility or technical limitations.3. **Complexity**: The content's complexity could justify a text file, requiring significant effort.4. **Purpose**: The project's intended format, like guides or manuals, could favor text files.5. **Collaboration**: Collaboration might have led to a text file for simplicity and ease of useFor the quality aspect:1. **Content**: Assess if the text is well-researched and organized.2. **Project Goals**: Reflect on whether the friend met key deliverables.3. **Exchange**: Discuss design decisions and challenges.4. **Supporting Materials**: Look at additional content for insight.In conclusion, the text file's format and content are more indicative of effort and quality than its mere format.Answer: \\boxed{Assistant 1}", ground_truth={"answer": [1]})
    output = reward(input)
    print(output)
