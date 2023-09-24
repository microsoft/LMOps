# Data Samples

## `input-raw-texts` Folder

- Each `*.txt` file in this folder represents a piece of raw text in the pre-training corpora.
- The title (if any) of each raw text is always the first sentence, separated from other contexts with the newline character `\n`.

## `output-read-compre` Folder

- This folder contains the output reading comprehension texts corresponding to the input raw texts.

## `domain.spm` and `general.spm`

- `general.spm`: The sentencepiece model of the general LLM (i.e., LLaMA-7B in our use case).
- `domain.spm`: The sentencepiece model trained from the target domain corpora, using the [sentencepiece tool](https://github.com/google/sentencepiece)
