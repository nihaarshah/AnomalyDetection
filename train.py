# Code from github.com/victoresque/pytorch-template
import argparse
import collections
import logging

import numpy as np
import torch

import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.VAE as VAE
from parse_config import ConfigParser
from trainer import Trainer

# fix random seeds for reproducibility
# SEED = 12344
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # np.random.seed(SEED)


def main(config):

    logger = config.get_logger("train")

    runs = config["seed_runs"]
    random_seed = True
    if config["seed"] != 0:
        runs = 1
        random_seed = False
        SEED = config["seed"]

    seed_results = []
    for _ in range(runs):
        if random_seed:
            SEED = np.random.randint(1e5)
            logger.info("SEED: {}".format(SEED))

        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # setup data_loader instances
        data_loader = config.init_obj("data_loader", module_data, data_type="train")
        valid_data_loader = None
        # valid_data_loader = config.init_obj("data_loader", module_data, data_type="val")
        test_data_loader = config.init_obj("data_loader", module_data, data_type="test")

        # build model architecture, then print to console
        model = config.init_obj("arch", VAE, input_size=data_loader.dataset.tensors[0].shape[1])
        # get function handles of loss and metrics
        criterion = getattr(module_loss, config["loss"])
        metrics = [getattr(module_metric, met) for met in config["metrics"]]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj("optimizer", torch.optim, trainable_params)

        lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            data_loader=data_loader,
            valid_data_loader=valid_data_loader,
            test_data_loader=test_data_loader,
            lr_scheduler=lr_scheduler,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        seed_results.append(eval_metrics["f1"])

    logger.info(seed_results)
    logger.info("Mean F1: {}".format(np.mean(seed_results)))
    logger.info("Std F1: {}".format(np.std(seed_results)))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c", "--config", default=None, type=str, help="config file path (default: None)"
    )
    args.add_argument(
        "-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
    )
    args.add_argument(
        "-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)"
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
    # logging.info(seed_results)
