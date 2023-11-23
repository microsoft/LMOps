# In-Context Demonstration Selection with Cross Entropy Difference

[Arxiv Link to Paper](https://arxiv.org/pdf/2305.14726.pdf)

Large language models (LLMs) can use in-context demonstrations to improve performance on zero-shot tasks. 
However, selecting the best in-context examples is challenging because model performance can vary widely depending on the selected examples. 
We present a cross-entropy difference (CED) method for selecting in-context demonstrations. 
Our method is based on the observation that the effectiveness of in-context demonstrations negatively
correlates with the perplexity of the test example by a language model that was finetuned
on that demonstration. 
We utilize parameter efficient finetuning to train small models on training data that are used for computing the cross-entropy difference between a test example and every candidate in-context demonstration. 
This metric is used to rank and select in-context demonstrations independently for each
test input. 
We evaluate our method on a mix-domain dataset that combines 8 benchmarks, representing 4 text generation tasks, showing that CED for in-context demonstration selection can improve performance for a variety of LLMs.

## Set up
Create a conda workspace named `cdsicd`. Use the requirements.txt to download dependencies. 

Commands for experimental setup are in runner.sh

This code base uses the [T-Few Module](https://github.com/r-three/t-few).

## Citation

```
@inproceedings{iter2023ced,  
      title={In-Context Demonstration Selection with Cross Entropy Difference},   
      author={Dan Iter and Reid Pryzant and Ruochen Xu and Shuohang Wang and Yang Liu and Yichong Xu and Chenguang Zhu},  
      year={2023},  
      booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing Findings",
      publisher = "Association for Computational Linguistics",
}
```
