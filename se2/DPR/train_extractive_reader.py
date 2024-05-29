#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import collections
import json
import sys

import hydra
import logging
import numpy as np
import os
import torch

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import List


from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import (
    ReaderSample,
    get_best_spans,
    SpanPrediction,
    ExtractiveReaderDataset,
)
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    set_cfg_params_from_state,
    get_encoder_params_state_from_cfg,
    setup_logger,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    get_model_file,
    setup_for_distributed_mode,
    get_model_obj,
)

logger = logging.getLogger()
setup_logger(logger)

ReaderQuestionPredictions = collections.namedtuple(
    "ReaderQuestionPredictions", ["id", "predictions", "gold_answers"]
)


class ReaderTrainer(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        model_file = get_model_file(self.cfg, self.cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, reader, optimizer = init_reader_components(
            cfg.encoder.encoder_model_type, cfg
        )

        reader, optimizer = setup_for_distributed_mode(
            reader,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self._load_saved_state(saved_state)

    def get_data_iterator(
        self,
        path: str,
        batch_size: int,
        is_train: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ) -> ShardedDataIterator:

        run_preprocessing = (
            True
            if self.distributed_factor == 1 or self.cfg.local_rank in [-1, 0]
            else False
        )

        gold_passages_src = self.cfg.gold_passages_src
        if gold_passages_src:
            if not is_train:
                gold_passages_src = self.cfg.gold_passages_src_dev

            assert os.path.exists(
                gold_passages_src
            ), "Please specify valid gold_passages_src/gold_passages_src_dev"

        dataset = ExtractiveReaderDataset(
            path,
            is_train,
            gold_passages_src,
            self.tensorizer,
            run_preprocessing,
            self.cfg.num_workers,
        )

        dataset.load_data()

        iterator = ShardedDataIterator(
            dataset,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
        )

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train_files,
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
        )

        # num_train_epochs = cfg.train.num_train_epochs - self.start_epoch

        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
            )
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        eval_step = cfg.train.eval_step
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        global_step = self.start_epoch * updates_per_epoch + self.start_batch

        for epoch in range(self.start_epoch, cfg.train.num_train_epochs):
            logger.info("***** Epoch %d *****", epoch)
            global_step = self._train_epoch(
                scheduler, epoch, eval_step, train_iterator, global_step
            )

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

        return

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # in distributed DDP mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]
        reader_validation_score = self.validate()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if reader_validation_score < (self.best_validation_result or 0):
                self.best_validation_result = reader_validation_score
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate(self):
        logger.info("Validation ...")
        cfg = self.cfg
        self.reader.eval()
        data_iterator = self.get_data_iterator(
            cfg.dev_files, cfg.train.dev_batch_size, False, shuffle=False
        )

        log_result_step = cfg.train.log_batch_step
        all_results = []

        eval_top_docs = cfg.eval_top_docs
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                cfg.passages_per_question_predict,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=False,
                shuffle=False,
            )

            input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            with torch.no_grad():
                start_logits, end_logits, relevance_logits = self.reader(
                    input.input_ids, attn_mask
                )

            batch_predictions = self._get_best_prediction(
                start_logits,
                end_logits,
                relevance_logits,
                samples_batch,
                passage_thresholds=eval_top_docs,
            )

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info("Eval step: %d ", i)

        ems = defaultdict(list)

        for q_predictions in all_results:
            gold_answers = q_predictions.gold_answers
            span_predictions = (
                q_predictions.predictions
            )  # {top docs threshold -> SpanPrediction()}
            for (n, span_prediction) in span_predictions.items():
                em_hit = max(
                    [
                        exact_match_score(span_prediction.prediction_text, ga)
                        for ga in gold_answers
                    ]
                )
                ems[n].append(em_hit)
        em = 0
        for n in sorted(ems.keys()):
            em = np.mean(ems[n])
            logger.info("n=%d\tEM %.2f" % (n, em * 100))

        if cfg.prediction_results_file:
            self._save_predictions(cfg.prediction_results_file, all_results)

        return em

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: ShardedDataIterator,
        global_step: int,
    ):
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step

        self.reader.train()
        epoch_batches = train_data_iterator.max_iterations

        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):

            data_iteration = train_data_iterator.get_iteration()

            # enables to resume to exactly same train state
            if cfg.fully_resumable:
                np.random.seed(cfg.seed + global_step)
                torch.manual_seed(cfg.seed + global_step)
                if cfg.n_gpu > 0:
                    torch.cuda.manual_seed_all(cfg.seed + global_step)

            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                cfg.passages_per_question,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=True,
                shuffle=True,
            )

            loss = self._calc_loss(input)

            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            max_grad_norm = cfg.train.max_grad_norm
            if cfg.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), max_grad_norm
                    )
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.reader.parameters(), max_grad_norm
                    )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.reader.zero_grad()
                global_step += 1

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                experiment.log_metric("lr", lr)
                logger.info(
                    "Epoch: %d: Step: %d/%d, global_step=%d, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    global_step,
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if global_step % eval_step == 0:
                logger.info(
                    "Validation: Epoch: %d Step: %d/%d",
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.reader.train()

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        experiment.log_metric("epoch_loss", epoch_loss)

        
        return global_step

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.reader)
        cp = os.path.join(
            cfg.output_dir,
            cfg.checkpoint_file_name
            + "."
            + str(epoch)
            + ("." + str(offset) if offset > 0 else ""),
        )

        meta_params = get_encoder_params_state_from_cfg(cfg)

        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info("Loading model weights from saved state ...")
            model_to_load.load_state_dict(saved_state.model_dict)

        logger.info("Loading saved optimizer state ...")
        if saved_state.optimizer_dict:
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[ReaderSample],
        passage_thresholds: List[int] = None,
    ) -> List[ReaderQuestionPredictions]:

        cfg = self.cfg
        max_answer_length = cfg.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(
            relevance_logits,
            dim=1,
            descending=True,
        )

        batch_results = []
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                passage_idx = idxs[q, p].item()
                if (
                    passage_idx >= non_empty_passages_num
                ):  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]
                p_end_logits = end_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(
                    self.tensorizer,
                    p_start_logits,
                    p_end_logits,
                    ctx_ids,
                    max_answer_length,
                    passage_idx,
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10,
                )
                nbest.extend(best_spans)
                if len(nbest) > 0 and not passage_thresholds:
                    break

            if passage_thresholds:
                passage_rank_matches = {}
                for n in passage_thresholds:
                    curr_nbest = [pred for pred in nbest if pred.passage_index < n]
                    passage_rank_matches[n] = curr_nbest[0]
                predictions = passage_rank_matches
            else:
                if len(nbest) == 0:
                    predictions = {
                        passages_per_question: SpanPrediction("", -1, -1, -1, "")
                    }
                else:
                    predictions = {passages_per_question: nbest[0]}
            batch_results.append(
                ReaderQuestionPredictions(sample.question, predictions, sample.answers)
            )
        return batch_results

    def _calc_loss(self, input: ReaderBatch) -> torch.Tensor:
        cfg = self.cfg
        input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)
        questions_num, passages_per_question, _ = input.input_ids.size()

        if self.reader.training:
            # start_logits, end_logits, rank_logits = self.reader(input.input_ids, attn_mask)
            loss = self.reader(
                input.input_ids,
                attn_mask,
                input.start_positions,
                input.end_positions,
                input.answers_mask,
            )

        else:
            # TODO: remove?
            with torch.no_grad():
                start_logits, end_logits, rank_logits = self.reader(
                    input.input_ids, attn_mask
                )

            loss = compute_loss(
                input.start_positions,
                input.end_positions,
                input.answers_mask,
                start_logits,
                end_logits,
                rank_logits,
                questions_num,
                passages_per_question,
            )
        if cfg.n_gpu > 1:
            loss = loss.mean()
        if cfg.train.gradient_accumulation_steps > 1:
            loss = loss / cfg.train.gradient_accumulation_steps

        return loss

    def _save_predictions(
        self, out_file: str, prediction_results: List[ReaderQuestionPredictions]
    ):
        logger.info("Saving prediction results to  %s", out_file)
        with open(out_file, "w", encoding="utf-8") as output:
            save_results = []
            for r in prediction_results:
                save_results.append(
                    {
                        "question": r.id,
                        "gold_answers": r.gold_answers,
                        "predictions": [
                            {
                                "top_k": top_k,
                                "prediction": {
                                    "text": span_pred.prediction_text,
                                    "score": span_pred.span_score,
                                    "relevance_score": span_pred.relevance_score,
                                    "passage_idx": span_pred.passage_index,
                                    "passage": self.tensorizer.to_string(
                                        span_pred.passage_token_ids
                                    ),
                                },
                            }
                            for top_k, span_pred in r.predictions.items()
                        ],
                    }
                )
            output.write(json.dumps(save_results, indent=4) + "\n")


@hydra.main(config_path="conf", config_name="extractive_reader_train_cfg")
def main(cfg: DictConfig):

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = ReaderTrainer(cfg)

    if cfg.train_files is not None:
        trainer.run_train()
    elif cfg.dev_files:
        logger.info("No train files are specified. Run validation.")
        trainer.validate()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()
