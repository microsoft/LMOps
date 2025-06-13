# Custom Model Code for RewardBench

Many reward models released since ChatGPT require various levels of custom code to run them.
For chat templates, we support both Huggingface Tokenizers and LMSYS FastChat (see `rewardbench/chattemplates.py`).

Contributing your model lets others compare your results to their reward models more directly (for example on datasets outside of the benchmark) when compared to submitting a `json` of results directly. 
It also enables us to verify results.

To contribute your model, you need to do the following:
1. Contribute a model loading and/or a pipeline function for your model. Model loading should handle quantization and the implementation of the model's `forward` function. The pipeline should follow HuggingFace's style and take in raw un-tokenized text and return a scalar output (or something already supported by our code, such as the right classifier `dict`).
2. Add your model to the `REWARD_MODEL_CONFIGS` in `rewardbench/models/__init__.py`.
3. Test your model with the `--debug` flag on `run_rm.py` to make sure inference works.

Please include links in your source code to any original implementation.
We'll handle it from here!