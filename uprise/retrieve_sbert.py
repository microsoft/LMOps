import hydra.utils as hu 
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import tqdm
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import faiss
import numpy as np
import json
from DPR.dpr.utils.tasks import task_map, test_cluster_map, train_cluster_map,get_prompt_files
from DPR.dpr.utils.data_utils import read_data_from_json_files
from src.dataset_readers.indexer_dsr import IndexerDatasetReader
from transformers import AutoTokenizer
import os
import logging

logger = logging.getLogger(__name__)

class KNNFinder:
    def __init__(self, cfg) -> None:
        self.cfg=cfg
        self.cuda_device = cfg.cuda_device
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.model_name,cache_dir=cfg.cache_dir)
        self.model = hu.instantiate(cfg.model).to(self.cuda_device)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer,device = self.cuda_device)
        self.n_docs=cfg.n_docs
        self.prompt_setup_type=cfg.prompt_setup_type
    
    def get_prompt_loader(self):
        # prompt_pool
        if self.cfg.train_clusters is not None:
            prompt_pool_path = get_prompt_files(self.cfg.prompt_pool_path, self.cfg.train_clusters)
        else:
            prompt_pool_path = self.cfg.prompt_pool_path
        logger.info("prompt files: %s", prompt_pool_path)
        self.prompt_pool = read_data_from_json_files(prompt_pool_path)
        logger.info("prompt passages num : %d", len(self.prompt_pool))
        
        self.corpus=[{'instruction':self.format_prompt(entry)} for entry in self.prompt_pool]
        prompt_reader = IndexerDatasetReader(self.tokenizer, self.corpus)
        prompt_loader = DataLoader(prompt_reader, batch_size=self.cfg.batch_size, collate_fn=self.co)
        return prompt_loader

    def create_index(self):
        prompt_loader=self.get_prompt_loader()
        for entry in tqdm.tqdm(prompt_loader): 
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model(**entry)
            id_list = np.array([m['id'] for m in metadata])
            self.index.add_with_ids(res.cpu().detach().numpy(), id_list)
    
    def format_prompt(self,entry):
        task=task_map.cls_dic[entry['task_name']]()
        if self.prompt_setup_type=='q':
            prompt = task.get_question(entry)
        elif self.prompt_setup_type=='a':
            prompt= task.get_answer(entry)
        elif self.prompt_setup_type=='qa':
            prompt=task.get_question(entry)+' '+task.get_answer(entry)
        return prompt
    
    def forward(self):
        res_list = []
        for i,entry in enumerate(tqdm.tqdm(self.dataloader)):
            with torch.no_grad():
                res = self.model(**entry)
            res = res.cpu().detach().numpy()
            res_list.extend([{"res":r,"metadata":m} for r,m in  zip(res,entry['metadata'])])

        return res_list

    def search(self,entry):
        res = np.expand_dims(entry['res'],axis=0)
        scores, near_ids = self.index.search(res,self.n_docs)
        return near_ids[0],scores[0] # task dim 0 as we expanded dim before
    
    
    def _find(self):
        res_list = self.forward()
        cntx_post = {}
        for entry in tqdm.tqdm(res_list):
            id=entry['metadata']['id']
            ctx_ids,ctx_scores = self.search(entry)
            cntx_post[id]=(ctx_ids,ctx_scores)
        return cntx_post
            
def find(cfg):
    finder = KNNFinder(cfg)
    finder.create_index()
    task_name=cfg.task_name
    logger.info("search for %s", task_name)
    task=task_map.cls_dic[task_name]() 
    dataset=task.get_dataset(cache_dir=cfg.cache_dir) 
    get_question = task.get_question
    queries=[]
    for id, entry in enumerate(dataset):
        entry['id']=id
        question=get_question(entry) 
        queries.append({'instruction':question})
    dataset_reader = IndexerDatasetReader(finder.tokenizer, queries)
    finder.dataloader=DataLoader(dataset_reader, batch_size=cfg.batch_size, collate_fn=finder.co)
    cntx_post=finder._find()
    
    merged_data=[]
    for idx,data in enumerate(dataset):
        ctx_ids,ctx_scores=cntx_post[idx]
        merged_data.append( 
        {
            "instruction": queries[idx],
            "meta_data": data,
            "ctxs": [
                {
                    "prompt_pool_id": str(prompt_pool_id), 
                    "passage": finder.corpus[prompt_pool_id],
                    "score": str(ctx_scores[i]),
                    "meta_data":finder.prompt_pool[prompt_pool_id] 
                }
                for i,prompt_pool_id in enumerate(ctx_ids)
            ],
        }
        )
    with open(cfg.out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", cfg.out_file)


@hydra.main(config_path="configs",config_name="sbert_retriever")
def main(cfg):
    logger.info(cfg)
    os.makedirs(os.path.dirname(cfg.out_file), exist_ok=True)
    find(cfg)


if __name__ == "__main__":
    main()