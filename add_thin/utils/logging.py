import inspect
import logging

import pytorch_lightning as pl
import rich
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax


def get_logger():
    caller = inspect.stack()[1]
    module = inspect.getmodule(caller.frame)
    logger_name = None
    if module is not None:
        logger_name = module.__name__.split(".")[-1]
    return logging.getLogger(logger_name)


@rank_zero_only
def print_config(config: DictConfig) -> None:
    content = OmegaConf.to_yaml(config, resolve=True)
    rich.print(Syntax(content, "yaml"))


def count_params(model: nn.Module):
    return {
        "params-total": sum(p.numel() for p in model.parameters()),
        "params-trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "params-not-trainable": sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        ),
    }
