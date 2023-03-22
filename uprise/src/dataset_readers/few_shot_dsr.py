'''
for inference
'''
from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import numpy as np
import json
import re
import more_itertools
from src.utils.dataset_utils import pad2sameLen
from DPR.dpr.utils.tasks import task_map

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)

class FewShotDatasetReader(torch.utils.data.Dataset):

    def __init__(self, model_name,task_name,prompt_file,num_prompts=-1,prompt_id=-1,
                    n_tokens=1600,cache_dir=None,max_length=2048) -> None:
        self.task=task_map.cls_dic[task_name]()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir,model_max_length=max_length)
        if self.task.class_num==1: #text completion task
            self.tokenizer.padding_side = "left"

        #retreived prompt_file,
        with open(prompt_file) as f:
            self.prompts = json.load(f)
        
        self.num_prompts = num_prompts
        self.prompt_id = prompt_id
        self.n_tokens_in_prompt = n_tokens
        self.num_processes = 1
        self.process_index =  0
        
    def __getitem__(self, index):
        if self.task.class_num==1: # text completion question
            return self.text_to_instance_completion(self.prompts[index])
        else: # multiple choice question
            return self.text_to_instance_choice(self.prompts[index])

    def __len__(self):
        return len(self.prompts)

    
    def shard(self,accelerator):
        self.num_processes = accelerator.num_processes
        self.process_index =  accelerator.process_index
        self.prompts = list(more_itertools.distribute(accelerator.num_processes,self.prompts)[accelerator.process_index])

    def get_length(self, text):
        tokenized_example = self.tokenizer.encode_plus(text,truncation=False,return_tensors='pt')
        shape = tokenized_example.input_ids.squeeze().shape
        if len(shape)==0:
            return 1
        else:
            return int(shape[0])
    def format_prompt(self,entry):
        prompt_task=task_map.cls_dic[entry['task_name']]()
        qa=prompt_task.get_question(entry)+' '+prompt_task.get_answer(entry)
        return remove_double_space(qa)

    def get_fields(self, entry):
        example=entry['meta_data']
        input_strs=self.task.get_input_strs(example)
        answer_strs = self.task.get_answers(example)
        label = self.task.get_label(example)
        prompts_list= [self.format_prompt(p['meta_data']) for p in entry['ctxs']]
        lengths_list= [self.get_length(prompt) for prompt in prompts_list]
        return input_strs,answer_strs,label,prompts_list,lengths_list

    def text_to_instance_choice(self, entry: Dict[str, Any],):
        '''
        multiple choice
        '''
        questions, answers, label, prompts_list, lengths_list = self.get_fields(entry) 
        
        max_q_length=0
        max_a_length=0
        for i in range(len(questions)) :  
            max_q_length=max(max_q_length,self.get_length(remove_double_space(questions[i])))
            max_a_length=max(max_a_length,self.get_length(remove_double_space(answers[i])))
        
        max_prompts = np.searchsorted(np.cumsum(lengths_list),self.n_tokens_in_prompt-(max_q_length+max_a_length))
        if self.num_prompts>-1:
            max_prompts = min(self.num_prompts,max_prompts) 
        if self.prompt_id> -1 and self.num_prompts==1:
            trunc_prompts_list = [prompts_list[::-1][self.prompt_id]]
        else:
            trunc_prompts_list = prompts_list[:max_prompts][::-1]

        prompt_enc_text = " \n ".join(trunc_prompts_list)
        input_ids_list=[]
        input_atten_mask_list=[]
        input_loss_mask_list=[]

        example={} 
        example['enc_text'] = []
        example['enc_answer']=[]
        for i in range(len(questions)):
            if max_prompts==0:
                enc_text=remove_double_space(questions[i]+answers[i])
                example['enc_text'].append(remove_double_space(questions[i]).strip()) #remove trailing space after question
            else:
                enc_text = remove_double_space(prompt_enc_text+' \n '+ questions[i]+answers[i])
                example['enc_text'].append(remove_double_space(prompt_enc_text+' \n '+ questions[i]).strip())#remove trailing space after question
            tokenized_example = self.tokenizer.encode_plus(enc_text,truncation=False,return_tensors='pt',add_special_tokens=False)
            enc_answer=remove_double_space(answers[i])
            tokenized_answer = self.tokenizer.encode_plus(enc_answer,truncation=False,add_special_tokens=False,return_tensors='pt')
            
            answer_mask=tokenized_answer.attention_mask.squeeze() 
            if len(answer_mask.shape)==0:
                answer_mask=torch.tensor([1]).to(answer_mask)
            
            input_ids=tokenized_example.input_ids.squeeze() 
            input_atten_mask=tokenized_example.attention_mask.squeeze() 
            input_loss_mask=torch.nn.functional.pad(answer_mask,(input_ids.shape[-1]-answer_mask.shape[-1],0))
            
            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            input_loss_mask_list.append(input_loss_mask)
            
            example['enc_answer'].append(enc_answer)
        example['n_prompts'] = str(max_prompts)
        example['label']=label
        return{
                'input_ids': pad2sameLen(input_ids_list,pad_idx=self.tokenizer.pad_token_id),
                'input_atten_mask': pad2sameLen(input_atten_mask_list,pad_idx=0),
                'input_loss_mask':pad2sameLen(input_loss_mask_list,pad_idx=0),
                'labels':  torch.tensor([label]), 
                "metadata": example,
            } 	

    def text_to_instance_completion(self, entry: Dict[str, Any],index=-1):
        '''
        text completion
        '''
        questions, answers, label, prompts_list, lengths_list = self.get_fields(entry) 
        
        max_q_length=0
        max_a_length=0
        for i in range(len(questions)) :  
            max_q_length=max(max_q_length,self.get_length(remove_double_space(questions[i])))
            max_a_length=max(max_a_length,self.get_length(remove_double_space(answers[i])))
        
        max_prompts = np.searchsorted(np.cumsum(lengths_list),self.n_tokens_in_prompt-(max_q_length+max_a_length))
        if self.num_prompts>-1:
            max_prompts = min(self.num_prompts,max_prompts)
        if self.prompt_id> -1 and self.num_prompts==1:
            trunc_prompts_list = [prompts_list[::-1][self.prompt_id]]
        else:
            trunc_prompts_list = prompts_list[:max_prompts][::-1]

        prompt_enc_text = " \n ".join(trunc_prompts_list)
        input_ids_list=[]
        input_atten_mask_list=[]
        example={} 
        example['enc_text'] = []
        example['enc_answer']=[]
        for i in range(len(questions)):
            if max_prompts==0:
                enc_text=remove_double_space(questions[i]).strip()  #remove trailing space after question
                example['enc_text'].append(remove_double_space(questions[i]).strip()) #remove trailing space after question
            else:
                enc_text = remove_double_space(prompt_enc_text+' \n '+ questions[i]).strip()  #remove trailing space after question
                example['enc_text'].append(remove_double_space(prompt_enc_text+' \n '+ questions[i]).strip())#remove trailing space after question
            
            tokenized_example = self.tokenizer.encode_plus(enc_text,truncation=False,return_tensors='pt',add_special_tokens=False)
            enc_answer=remove_double_space(answers[i])
            
            input_ids=tokenized_example.input_ids.squeeze() 
            input_atten_mask=tokenized_example.attention_mask.squeeze() 
            
            if len(input_ids.shape)==0:
                input_ids=input_ids.unsqueeze(0)
                input_atten_mask=input_atten_mask.unsqueeze(0)
            
            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)

            example['enc_answer'].append(enc_answer)
        example['temp_label']=label
        example['n_prompts'] = str(max_prompts)
        example['label']=label
        return{
                'input_ids': pad2sameLen(input_ids_list,pad_idx=self.tokenizer.pad_token_id,left_pad=True),
                'input_atten_mask': pad2sameLen(input_atten_mask_list,pad_idx=0,left_pad=True),
                "metadata": example,
            }	                     	 		 	 	
                     	 		 	 	
                     	 	