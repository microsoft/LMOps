import collections
import torch
import numpy as np
import verl.utils.torch_functional as verl_F

from typing import Callable, List, Optional, Tuple, Mapping
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class ResponseArena:

  def __init__(self, config, online_transform_func: Callable, tokenizer: PreTrainedTokenizer, input_tokenizer: PreTrainedTokenizer, rank: Optional[int] = 0, prompt_length: int = 1024):
    self.config = config
    self.online_transform_func = online_transform_func
    self.tokenizer = tokenizer
    self.input_tokenizer = input_tokenizer
    self.seed = config.seed
    self.rand = np.random.RandomState(self.seed)
    self.rank = rank
    self.max_assistant_length = (prompt_length - 200) // 2
  
  def _remove_think(self, txt):
    from deepscaler.globals import  THOUGHT_DELIMITER_END
    # txt = txt.split(THOUGHT_DELIMITER_END)[-1]
    _split = txt.split(THOUGHT_DELIMITER_END)
    if len(_split) < 2:
       txt = "[NO ANSWER]"
    else:
      txt = _split[-1]
    return txt
  
  def _truncate_response(self, response: str):
    # we tokenize the response first
    response_ids = self.input_tokenizer.encode(response, add_special_tokens=False)
    if len(response_ids) > self.max_assistant_length:
      # keep the last max_assistant_length tokens
      response_ids = response_ids[-self.max_assistant_length:]
      response = self.input_tokenizer.decode(response_ids, skip_special_tokens=True)
    return response
    
  def apply_arena_template(self, query, response, arena_bsl_response, do_shuffle=True):
    """
    Apply the arena template to the query and a pair of responses.
    """
    do_shuffle_response = (self.rand.rand() < 0.5) and do_shuffle
    if do_shuffle_response:
        resp1 = arena_bsl_response
        resp2 = response
    else:
        resp1 = response
        resp2 = arena_bsl_response
    
    if self.config.get("remove_think", True):
      resp1 = self._remove_think(resp1)
      resp2 = self._remove_think(resp2)

    # TODO maybe truncate each response
    resp1 = self._truncate_response(resp1)
    resp2 = self._truncate_response(resp2)
    
    transform_input = {
        "question": query,
        "response1": resp1,
        "response2": resp2,
        "answer" : "1",
        "extra_info": None,
        "data_source": "_all",
    }
    output = self.online_transform_func(transform_input)
    chat = output['prompt']

    prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    return prompt_with_chat_template, do_shuffle_response

  def map_data_to_arena_matches(self, data: DataProto):
    """
    Map the data to arena matches.
    """
    raise NotImplementedError("This method should be implemented in a subclass.")

  def reduce_aerna_match_results_to_rewards(self, data: DataProto, all_matches: List[Tuple[int, int]]):
    """
    Reduce the arena match results to rewards.
    """
    raise NotImplementedError("This method should be implemented in a subclass.")
  

class ResponseArenaSimple(ResponseArena):

  def _response_batch_to_str(self, data: DataProto, data_idx: int, tok):
    response_ids = data.batch['responses'][data_idx]
    response_length = response_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][data_idx][-response_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    response = tok.decode(valid_response_ids)
    response = response.replace(tok.eos_token, '')
    return response

  def map_data_to_arena_matches(self, data: DataProto):
    """
    Map the data to arena matches.
    """
    src_max_length = data.batch['attention_mask'].shape[-1]

    # NOTE: data are not necessarily with the same query
    n_arena_bsl = self.config.get('n_arena_bsl', 4)
    arena_group_size = self.config.arena_group_size
    assert data.batch.batch_size[0] % arena_group_size == 0, f"batch size {data.batch.batch_size[0]} should be divisible by arena group size {arena_group_size}"
    arena_n_group = data.batch.batch_size[0] // arena_group_size

    query_indices = [None] * arena_n_group
    all_prompts = []
    all_matches = []
    all_input_ids = []
    all_attention_masks = []

    # remove debug
    # for i in range(data.batch.batch_size[0]):
    #     # print all the queries
    #     chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()
    #     print(f"chat {i}: {chat}", flush=True)

    for group_idx in range(arena_n_group):
        # check whether the queries in the group are the same

        arena_matches = []
        arena_bsl_responses = []
        query = None


        for i in range(arena_group_size):
            data_idx = group_idx * arena_group_size + i
            # query_index = data.batch['index'][data_idx].item()
            query_index = data.non_tensor_batch['uid'][data_idx]
            # print(f"group_idx: {group_idx}, data_idx: {data_idx}, query_index: {query_index}", flush=True)
            if query_indices[group_idx] is None:
                query_indices[group_idx] = query_index
            else:
                assert query_index == query_indices[group_idx], f"query index {query_index} should be the same for group {group_idx}, but got {query_indices[group_idx]}; query={query} current query={data.non_tensor_batch['raw_prompt'][i].tolist()[-1]['content']}"
            
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()
            assert len(chat) <= 2 and chat[-1]["role"] == "user", f"chat {chat} should be a list of length 2 or 1 and the last role should be user (aka the user query)"

            if query is None:
                query = chat[-1]["content"]
            
            response = self._response_batch_to_str(data, data_idx, self.input_tokenizer)

            # if i < n_arena_bsl:
            #     arena_bsl_responses.append(response)
            
            def _tokenize(text):
                max_length = self.config.get('max_length', src_max_length)
                if max_length is None:
                    max_length = src_max_length
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                    prompt=text,
                    tokenizer=self.tokenizer,
                    max_length=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,  # right padding
                    truncation=self.config.get('truncation', 'right'))  # truncate from the right

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
            
            for j in range(n_arena_bsl):
                if j >= len(arena_bsl_responses):
                    break
                prompt_with_chat_template, responses_shuffled = self.apply_arena_template(query, response, arena_bsl_responses[j])
                all_prompts.append(prompt_with_chat_template)
                _tokenize(prompt_with_chat_template)
                if responses_shuffled:
                    arena_matches.append((j, i))
                else:
                    arena_matches.append((i, j))
            
            # append current response at the end of the loop to avoid self comparison
            if i < n_arena_bsl:
                arena_bsl_responses.append(response)
        all_matches.append(arena_matches)

    if self.rank == 0:
        print(f"[INFO] all_prompts[0]: {all_prompts[0]}", flush=True)
    rm_input_ids = torch.cat(all_input_ids, dim=0)
    rm_attention_mask = torch.cat(all_attention_masks, dim=0)

    rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

    rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

    ret = DataProto.from_dict(tensors=rm_inputs, meta_info={'eos_token_id': self.tokenizer.eos_token_id, 'do_sample': True})
    return ret, all_matches
  
  def _extract_match_winner(self, response: str, invalid_judge_metric: Mapping):
    from deepscaler.rewards.judge_extractor import extract_judge
    return extract_judge(response, invalid_judge_metric)
  
  def reduce_aerna_match_results_to_rewards(self, data: DataProto, all_matches: List[Tuple[int, int]]):
    """
    Reduce the arena match results to rewards.
    """
    assert data.batch.batch_size[0] == len(all_matches) * len(all_matches[0]), f"batch size {data.batch.batch_size[0]} should be equal to {len(all_matches)} * {len(all_matches[0])}"

    n_arena_bsl = self.config.get('n_arena_bsl', 4)
    arena_group_size = self.config.arena_group_size

    data_idx = 0 
    all_scores = []
    invalid_judge_metric = collections.defaultdict(float)
    for group_idx, arena_matches in enumerate(all_matches):
        
        # the bsl responses are assigned with 0.5 because they are not compared with themselves
        scores = [0.5] * n_arena_bsl + [0] * (arena_group_size - n_arena_bsl)
        
        for player_idx1, player_idx2 in arena_matches:
            assert player_idx1 < n_arena_bsl or player_idx2 < n_arena_bsl, f"player index {player_idx1} and {player_idx2} should be less than {n_arena_bsl}, which plays the role of arena bsl"
            response = self._response_batch_to_str(data, data_idx, self.tokenizer)
            player1_win = self._extract_match_winner(response, invalid_judge_metric)
            if player_idx1 < n_arena_bsl:
                if not player1_win:
                    scores[player_idx2] += 1.0
            # NOTE player1 and player2 could be both arena bsl so one of them must get scores, so here we do not use 'elif'
            if player_idx2 < n_arena_bsl:
                if player1_win:
                    scores[player_idx1] += 1.0
                
            data_idx += 1
        
        all_scores.extend(scores)
    
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    # normalize the scores
    all_scores = all_scores * 0.5 - 1.0

    print(f"[INFO] invalid_judge_metric: {invalid_judge_metric}", flush=True)
    return all_scores


class ResponseArenaElo(ResponseArena):
  
  def __init__(self, config, online_transform_func: Callable, tokenizer: PreTrainedTokenizer, input_tokenizer: PreTrainedTokenizer, rank: Optional[int] = 0, prompt_length: int = 1024):
    super().__init__(config, online_transform_func, tokenizer, input_tokenizer, rank, prompt_length)
    # there will be elo_match_multiplier * arena_group_size matches for each group
    self.elo_match_multiplier = self.config.get('elo_match_multiplier', 4)
    self.arena_group_size = self.config.arena_group_size
    assert self.elo_match_multiplier < self.arena_group_size, f"elo_match_multiplier {self.elo_match_multiplier} should be less than arena_group_size {self.arena_group_size}"
    self.arena_num_group_matches = self.elo_match_multiplier * self.arena_group_size
    from evalica import elo, Winner
    self.elo = elo
    self.judge2winner = [Winner.Draw, Winner.X, Winner.Y]
  
  @staticmethod
  def _response_batch_to_str(data: DataProto, data_idx: int, tok):
    response_ids = data.batch['responses'][data_idx]
    response_length = response_ids.shape[-1]
    valid_response_length = data.batch['attention_mask'][data_idx][-response_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    response = tok.decode(valid_response_ids)
    response = response.replace(tok.eos_token, '')
    return response

  @staticmethod
  def grouped_matches_to_2d_array(all_matches: List[List[Tuple[int, int]]]):
    """
    Convert the grouped matches to a 2D tensor with shape: (n_group, self.elo_match_multiplier, 2)
    """
    all_matches_2d = []
    for group_idx, arena_matches in enumerate(all_matches):
      for match_idx, (p1, p2) in enumerate(arena_matches):
        all_matches_2d.append([p1, p2, group_idx])
    
    all_matches_2d = np.array(all_matches_2d, dtype=object)
    return all_matches_2d

  @staticmethod
  def _2d_tensor_to_grouped_matches(all_matches_2d):
    """
    Convert the 2D tensor to grouped matches.
    """
    all_matches = []
    for i in range(all_matches_2d.shape[0]):
      p1, p2, group_idx = all_matches_2d[i].tolist()
      while len(all_matches) <= group_idx:
        all_matches.append([])
      all_matches[group_idx].append((p1, p2))
    
    return all_matches

  def map_data_to_arena_matches(self, data: DataProto):
    """
    Map the data to arena matches.
    """
    src_max_length = data.batch['attention_mask'].shape[-1]

    # NOTE: data are not necessarily with the same query
    arena_group_size = self.arena_group_size
    assert data.batch.batch_size[0] % arena_group_size == 0, f"batch size {data.batch.batch_size[0]} should be divisible by arena group size {arena_group_size}"
    arena_n_group = data.batch.batch_size[0] // arena_group_size

    query_indices = [None] * arena_n_group
    all_prompts = []
    all_matches = []
    all_input_ids = []
    all_attention_masks = []
    all_uids = []
    all_has_labels = []

    for group_idx in range(arena_n_group):
      arena_matches = []
      group_responses = []
      query = None

      for i in range(arena_group_size):
        data_idx = group_idx * arena_group_size + i
        # query_index = data.batch['index'][data_idx].item()
        query_index = data.non_tensor_batch['uid'][data_idx]
        # print(f"group_idx: {group_idx}, data_idx: {data_idx}, query_index: {query_index}", flush=True)
        if query_indices[group_idx] is None:
            query_indices[group_idx] = query_index
        else:
            assert query_index == query_indices[group_idx], f"query index {query_index} should be the same for group {group_idx}, but got {query_indices[group_idx]}; query={query} current query={data.non_tensor_batch['raw_prompt'][i].tolist()[-1]['content']}"
        
        chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()
        assert len(chat) <= 2 and chat[-1]["role"] == "user", f"chat {chat} should be a list of length 2 or 1 and the last role should be user (aka the user query)"

        if query is None:
            query = chat[-1]["content"]
        
        response = self._response_batch_to_str(data, data_idx, self.input_tokenizer)
        group_responses.append(response)

        # if i < n_arena_bsl:
        #     arena_bsl_responses.append(response)
      for i in range(arena_group_size):
        data_idx = group_idx * arena_group_size + i
        def _tokenize(text):
          max_length = self.config.get('max_length', src_max_length)
          if max_length is None:
            max_length = src_max_length
          input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=text,
            tokenizer=self.tokenizer,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,  # right padding
            truncation=self.config.get('truncation', 'right'))  # truncate from the right

          all_input_ids.append(input_ids)
          all_attention_masks.append(attention_mask)
          all_uids.append(query_indices[group_idx])
        
        # i in the opponent set to avoid self comparison
        _opponent_set = set([i])
        while len(_opponent_set) <= self.elo_match_multiplier:
          # select a random opponent
          opponent_idx = self.rand.randint(0, arena_group_size)
          if opponent_idx not in _opponent_set:
            _opponent_set.add(opponent_idx)
            prompt_with_chat_template, responses_shuffled = self.apply_arena_template(query, group_responses[i], group_responses[opponent_idx])
            all_prompts.append(prompt_with_chat_template)
            _tokenize(prompt_with_chat_template)
            has_label = True
            data_item = data[data_idx]
            if data_item.non_tensor_batch['reward_model']['ground_truth'] is None or data_item.non_tensor_batch['reward_model']['ground_truth'] == '' or data_item.non_tensor_batch['reward_model']['style'] == 'model':
              has_label = False
            all_has_labels.append(has_label)
            if responses_shuffled:
              arena_matches.append((opponent_idx, i))
            else:
              arena_matches.append((i, opponent_idx))
          

      all_matches.append(arena_matches)
    assert len(all_input_ids) == self.elo_match_multiplier * data.batch.batch_size[0] == self.arena_num_group_matches * arena_n_group, f"len(all_input_ids) {len(all_input_ids)} should be equal to {self.elo_match_multiplier} * {data.batch.batch_size[0]} == {self.arena_num_group_matches} * {arena_n_group}"
    
    if self.rank == 0:
        print(f"[INFO] all_prompts[0]: {all_prompts[0]}", flush=True)
    rm_input_ids = torch.cat(all_input_ids, dim=0)
    rm_attention_mask = torch.cat(all_attention_masks, dim=0)

    rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
    all_has_labels = torch.tensor(all_has_labels, dtype=torch.bool)

    rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids, 'has_labels': all_has_labels}

    ret = DataProto.from_dict(tensors=rm_inputs, meta_info={'eos_token_id': self.tokenizer.eos_token_id, 'do_sample': True})
    ret.non_tensor_batch['uid'] = np.array(all_uids, dtype=object)
    return ret, all_matches

  def _extract_match_winner(self, response: str, invalid_judge_metric: Mapping):
    from deepscaler.rewards.judge_extractor import extract_judge_for_elo
    _judge = extract_judge_for_elo(response, invalid_judge_metric)
    return self.judge2winner[_judge]
  
  def reduce_aerna_match_results_to_rewards(self, data: DataProto, all_matches: List[Tuple[int, int]]):
    """
    Reduce the arena match results to rewards.
    """
    assert data.batch.batch_size[0] == len(all_matches) * len(all_matches[0]), f"batch size {data.batch.batch_size[0]} should be equal to {len(all_matches)} * {len(all_matches[0])}"

    all_scores = []
    invalid_judge_metric = collections.defaultdict(float)
    for group_idx, arena_matches in enumerate(all_matches):
      player_x = []
      player_y = []
      match_results = []
      for match_idx, (p1, p2) in enumerate(arena_matches):
        player_x.append(p1)
        player_y.append(p2)
        data_idx = group_idx * self.arena_num_group_matches + match_idx
        response = self._response_batch_to_str(data, data_idx, self.tokenizer)
        match_results.append(self._extract_match_winner(response, invalid_judge_metric))
      
      # calculate elo scores
      elo_scores = self.elo(player_x, player_y, match_results).scores
      for pid in range(self.arena_group_size):
        all_scores.append(float(elo_scores[pid]))
    
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    all_scores = (all_scores - 1000 ) / 200 + 0.5
    print(f"[INFO] invalid_judge_metric: {invalid_judge_metric}", flush=True)
    return all_scores


class ResponseArenaFast(ResponseArena):
   
  def __init__(self, config, online_transform_func: Callable, tokenizer: PreTrainedTokenizer, input_tokenizer: PreTrainedTokenizer, rank: Optional[int] = 0, prompt_length: int = 1024):
    super().__init__(config, online_transform_func, tokenizer, input_tokenizer, rank, prompt_length)
    self.arena_group_size = self.config.arena_group_size
    assert self.arena_group_size > 1, f"arena_group_size {self.arena_group_size} should be greater than 1"
    self.arena_num_group_matches = self.arena_group_size
  
  def map_data_to_arena_matches(self, data: DataProto):
    src_max_length = data.batch['attention_mask'].shape[-1]

    arena_group_size = self.arena_group_size
    assert data.batch.batch_size[0] % arena_group_size == 0, f"batch size {data.batch.batch_size[0]} should be divisible by arena group size {arena_group_size}"
    arena_n_group = data.batch.batch_size[0] // arena_group_size

    # debug: print all group query uids
    for group_idx in range(arena_n_group):
       for i in range(arena_group_size):
          data_idx = group_idx * arena_group_size + i
          query_index = data.non_tensor_batch['uid'][data_idx]
          print(f"group_idx: {group_idx}, data_idx: {data_idx}, query_index: {query_index}", flush=True)        

    query_indices = [None] * arena_n_group
    all_prompts = []
    all_matches = []
    all_input_ids = []
    all_attention_masks = []
    all_uids = []
    all_has_labels = []

    for group_idx in range(arena_n_group):
      arena_matches = []
      group_responses = []
      query = None
  
      for i in range(arena_group_size):
        data_idx = group_idx * arena_group_size + i
        # query_index = data.batch['index'][data_idx].item()
        query_index = data.non_tensor_batch['uid'][data_idx]
        # print(f"group_idx: {group_idx}, data_idx: {data_idx}, query_index: {query_index}", flush=True)
        if query_indices[group_idx] is None:
          query_indices[group_idx] = query_index
        else:
          assert query_index == query_indices[group_idx], f"query index {query_index} should be the same for group {group_idx}, but got {query_indices[group_idx]}; query={query} current query={data.non_tensor_batch['raw_prompt'][i].tolist()[-1]['content']}"
        
        chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()
        assert len(chat) <= 2 and chat[-1]["role"] == "user", f"chat {chat} should be a list of length 2 or 1 and the last role should be user (aka the user query)"

        if query is None:
          query = chat[-1]["content"]
        
        response = ResponseArenaElo._response_batch_to_str(data, data_idx, self.input_tokenizer)
        group_responses.append(response)
      
      # randomly select an opponent (aka baseline) from the group
      if group_idx % 2 == 1:
        do_swap_candidate = True
      else:
        do_swap_candidate = False 
      _opponent = self.rand.randint(0, arena_group_size)
      for i in range(arena_group_size):
        data_idx = group_idx * arena_group_size + i
        def _tokenize(text):
          # max_length = self.config.get('max_length', src_max_length)
          # if max_length is None:
          #   max_length = src_max_length
          max_length = self.config.max_length
          input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=text,
            tokenizer=self.tokenizer,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,  # right padding
            truncation=self.config.get('truncation', 'right'))
        
          all_input_ids.append(input_ids)
          all_attention_masks.append(attention_mask)
          all_uids.append(query_indices[group_idx])
        
        resp0 = group_responses[i]
        resp1 = group_responses[_opponent]
        if do_swap_candidate:
          resp0, resp1 = resp1, resp0
        prompt_with_chat_template, responses_shuffled = self.apply_arena_template(query, resp0, resp1, do_shuffle=False)
        assert not responses_shuffled
        all_prompts.append(prompt_with_chat_template)
        _tokenize(prompt_with_chat_template)

        if self.rank == 0 and len(all_input_ids) == 1:
          # print(f"all_input_ids[0]={all_input_ids[0]}", flush=True)
          decoded_prompt = self.tokenizer.decode(all_input_ids[0][0], skip_special_tokens=True)
          print(f"[INFO] decoded_prompt: {decoded_prompt}", flush=True)

        has_label = True
        data_item = data[data_idx]
        if data_item.non_tensor_batch['reward_model']['ground_truth'] is None or data_item.non_tensor_batch['reward_model']['ground_truth'] == '' or data_item.non_tensor_batch['reward_model']['style'] == 'model':
          has_label = False
        all_has_labels.append(has_label)
        if do_swap_candidate:
           arena_matches.append((_opponent, i))
        else:
          arena_matches.append((i, _opponent))
    
      all_matches.append(arena_matches)
    
    assert len(all_input_ids) == data.batch.batch_size[0] == self.arena_num_group_matches * arena_n_group, f"len(all_input_ids) {len(all_input_ids)} should be equal to {data.batch.batch_size[0]} == {self.arena_num_group_matches} * {arena_n_group}"

    if self.rank == 0:
      print(f"[INFO] all_prompts[0]: {all_prompts[0]}", flush=True)
    
    rm_input_ids = torch.cat(all_input_ids, dim=0)
    rm_attention_mask = torch.cat(all_attention_masks, dim=0)

    rm_position_ids = compute_position_id_with_mask(rm_attention_mask)
    all_has_labels = torch.tensor(all_has_labels, dtype=torch.bool)

    rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids, 'has_labels': all_has_labels}

    ret = DataProto.from_dict(tensors=rm_inputs, meta_info={'eos_token_id': self.tokenizer.eos_token_id, 'do_sample': True})
    ret.non_tensor_batch['uid'] = np.array(all_uids, dtype=object)
    return ret, all_matches

  def reduce_aerna_match_results_to_rewards(self, data: DataProto, all_matches: List[Tuple[int, int]]):
    from deepscaler.rewards.judge_extractor import extract_judge
    """
    Reduce the arena match results to rewards.
    """
    assert data.batch.batch_size[0] == len(all_matches) * len(all_matches[0]), f"batch size {data.batch.batch_size[0]} should be equal to {len(all_matches)} * {len(all_matches[0])}"

    all_scores = []
    invalid_judge_metric = collections.defaultdict(float)
    for group_idx, arena_matches in enumerate(all_matches):
      do_swap_candidate = (group_idx % 2 == 1)
      for match_idx, (p1, p2) in enumerate(arena_matches):
        data_idx = group_idx * self.arena_num_group_matches + match_idx
        response = ResponseArenaElo._response_batch_to_str(data, data_idx, self.tokenizer)
        judge = extract_judge(response, invalid_judge_metric)
        if judge == 1:
          all_scores.append(0.0 if do_swap_candidate else 1.0)
        elif judge == 2:
          all_scores.append(1.0 if do_swap_candidate else 0.0)
        elif judge == -1:
          all_scores.append(0)
        else:
           raise ValueError(f"Invalid judge value {judge} extracted from response: {response}")

    
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    print(f"[INFO] invalid_judge_metric: {invalid_judge_metric}", flush=True)
    return all_scores

  @staticmethod
  def grouped_matches_to_2d_array(all_matches: List[List[Tuple[int, int]]]):
    return ResponseArenaElo.grouped_matches_to_2d_array(all_matches)

  @staticmethod
  def _2d_tensor_to_grouped_matches(all_matches_2d):
    return ResponseArenaElo._2d_tensor_to_grouped_matches(all_matches_2d)

    