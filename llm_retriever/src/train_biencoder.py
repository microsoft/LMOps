import logging

from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast
)

from logger_config import logger, LoggerCallback
from config import Arguments
from trainers import BiencoderTrainer
from loaders import RetrievalDataLoader
from collators import BiencoderCollator
from models import BiencoderModel


def _common_setup(args: Arguments):
    set_verbosity_info()
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
        set_verbosity_warning()
    enable_explicit_format()
    set_seed(args.seed)


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: BiencoderModel = BiencoderModel.build(args=args)
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = BiencoderCollator(
        args=args,
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)

    retrieval_data_loader = RetrievalDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = retrieval_data_loader.train_dataset

    trainer: Trainer = BiencoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    retrieval_data_loader.set_trainer(trainer)
    model.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    return


if __name__ == "__main__":
    main()
