import collections
import torch
import numpy as np
import verl.utils.torch_functional as verl_F

from typing import Callable, List, Optional, Tuple, Mapping
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from deepscaler.rewards.math_utils.utils import extract_answer


class GeneralVerifier:

  THOUGHT_DELIMITER_END = "</think>"
  VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    "If the student's answer is correct, output \"Final Decision: Yes\". If the student's answer is incorrect, output \"Final Decision: No\". Assistant:"
  )
  VERIFIER_PASS_TAG = "Final Decision: Yes"

  def __init__(self, config, tokenizer: PreTrainedTokenizer, input_tokenizer: PreTrainedTokenizer, rank: Optional[int] = 0, max_answer_length: int = 256):
    self.config = config
    self.tokenizer = tokenizer
    self.input_tokenizer = input_tokenizer
    self.seed = config.seed
    self.rand = np.random.RandomState(self.seed)
    self.rank = rank
    self.max_answer_length = max_answer_length
  
  def _truncate_student_answer(self, student_answer: str) -> Tuple[str, List[int]]:
    ans_ids = self.input_tokenizer.encode(student_answer, add_special_tokens=False)
    if len(ans_ids) > self.max_answer_length:
      # keep the last max_assistant_length tokens
      ans_ids = ans_ids[-self.max_answer_length:]
      student_answer = self.input_tokenizer.decode(ans_ids, skip_special_tokens=True)
    return student_answer, ans_ids
  
  def apply_verify_query_template(self, question: str, teacher_answer: str, student_answer: str) -> str:
    """
    Applies the verify query template to the question, teacher answer, and student answer.
    """
    query = self.VERIFIER_PROMPT_TEMPLATE.format(
      question=question,
      ground_truth=teacher_answer,
      student_answer=student_answer
    )
    return query
  
  def extract_answer_from_response(self, response: str) -> str:
    model_solution = response.split(self.THOUGHT_DELIMITER_END)[-1]
    try:
      model_answer = extract_answer(model_solution)
    except Exception as e:
      print(f"Error extracting answer: {e}")
      model_answer = "No answer found"
    if model_answer is None:
      model_answer = "No answer found"
    return model_answer.strip()

  def _response_batch_to_str(self, data: DataProto, data_idx: int, tok):
    response_ids = data.batch['responses'][data_idx]
    response_length = response_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][data_idx][-response_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    response = tok.decode(valid_response_ids)
    response = response.replace(tok.eos_token, '')
    return response
  
  def map_data_to_verify_query(self, data: DataProto):
    """
    Maps the data to a verify query format.
    """

    all_input_ids = []
    all_attention_mask = []
    all_has_labels = []
    all_other_info = []
    for i in range(len(data)):
      chat: list = data[i].non_tensor_batch['raw_prompt'].tolist()
      assert len(chat) <= 2 and chat[-1]["role"] == "user", f"chat {chat} should be a list of length 2 or 1 and the last role should be user (aka the user query)"
      question = chat[-1]["content"]

      response = self._response_batch_to_str(data, i, self.input_tokenizer)
      student_answer = self.extract_answer_from_response(response)
      student_answer, stu_ans_ids = self._truncate_student_answer(student_answer)
      teacher_answer = data[i].non_tensor_batch['reward_model']['ground_truth']

      tea_ans_ids = self.input_tokenizer.encode(teacher_answer, add_special_tokens=False)

      query = self.apply_verify_query_template(question, teacher_answer, student_answer)
      input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=query,
        tokenizer=self.tokenizer,
        max_length=4096,
        pad_token_id=self.tokenizer.pad_token_id,
        left_pad=True,  # right padding
        truncation=self.config.get('truncation', 'left'))

      all_input_ids.append(input_ids)
      all_attention_mask.append(attention_mask)
      all_other_info.append((stu_ans_ids, tea_ans_ids))

      has_label = True
      data_item = data[i]
      if data_item.non_tensor_batch['reward_model']['ground_truth'] is None or data_item.non_tensor_batch['reward_model']['ground_truth'] == '' or data_item.non_tensor_batch['reward_model']['style'] == 'model':
        has_label = False
      all_has_labels.append(has_label)
    
      if self.rank == 0 and i == 0:
        print(f"Verify query for data index {i}: {query}")
    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_mask = torch.cat(all_attention_mask, dim=0)
    all_position_ids = compute_position_id_with_mask(all_attention_mask)
    all_has_labels = torch.tensor(all_has_labels, dtype=torch.bool)

    verifiy_inputs = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'position_ids': all_position_ids, 'has_labels': all_has_labels}

    ret = DataProto.from_dict(tensors=verifiy_inputs, meta_info={'eos_token_id': self.tokenizer.eos_token_id, 'do_sample': True})
    return ret, all_other_info

  def reduce_verify_query_results_to_scores(self, data: DataProto, all_other_info: List[Tuple[List[int], List[int]]]) -> torch.Tensor:
    assert len(data) == len(all_other_info), "data and all_other_info must have the same length"
    all_scores = []
    for i in range(len(data)):
      stu_ans_ids, tea_ans_ids = all_other_info[i]
      response = self._response_batch_to_str(data, i, self.tokenizer)
      score = 0.0
      if self.VERIFIER_PASS_TAG in response:
        score = 1.0
        difference = abs(len(stu_ans_ids) - len(tea_ans_ids))
        difference = min(difference, 10)  
        score -= difference * 0.05
      
      all_scores.append(score)
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    return all_scores
