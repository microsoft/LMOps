# Data Samples

## `input-raw-texts` Folder

- Each `*.txt` file in this folder represents a piece of raw text in the pre-training corpora.
- The title (if any) of each raw text is always the first sentence, separated from other contexts with the newline character `\n`.

## `output-read-compre` Folder

- This folder contains the output reading comprehension texts corresponding to the input raw texts.
- **NOTE:** If you concatenate multiple data examples together as a single input for the language model, please wrap each reading comprehension text using templates like this: `{"text": "READCOMPRE"}`, where `READCOMPRE` is one piece of reading comprehension text. This format resembles the format of a dictionary, explicitly separating each reading comprehension text from others. This allows the model to focus on the context of the current text when training on comprehension tasks.

## `domain.spm` and `general.spm`

- `general.spm`: The sentencepiece model of the general LLM (i.e., LLaMA-7B in our use case).
- `domain.spm`: The sentencepiece model trained from the target domain corpora, using the [sentencepiece tool](https://github.com/google/sentencepiece)