import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        datasets = cfg.datasets # encoder_train_default_test
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        # print(datasets)
        if self.train_datasets_names:
            self.train_datasets = [
            ]
            for ds_name in self.train_datasets_names:
                datasets[ds_name]['loss_type'] = cfg.loss_type
                tmp_dataset = hydra.utils.instantiate(datasets[ds_name])
                self.train_datasets.append(tmp_dataset)
        else:
            self.train_datasets = []
        if cfg.dev_datasets:
            self.dev_datasets_names = cfg.dev_datasets
            logger.info("dev_datasets: %s", self.dev_datasets_names)
            self.dev_datasets = [
                hydra.utils.instantiate(datasets[ds_name])
                for ds_name in self.dev_datasets_names
            ]
        self.sampling_rates = cfg.train_sampling_rates
