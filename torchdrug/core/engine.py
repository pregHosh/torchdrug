import copy
import logging
import os
import sys
from itertools import islice
from typing import List, Optional, Union

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import core, data, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty

module = sys.modules[__name__]
logger = logging.getLogger(__name__)


@R.register("core.Engine")
class Engine(core.Configurable):
    """
    General class that handles everything about training and test of a task.

    This class can perform synchronous distributed parallel training over multiple CPUs or GPUs.
    To invoke parallel training, launch with one of the following commands.

    1. Single-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    2. Multi-node multi-process case.

    .. code-block:: bash

        python -m torch.distributed.launch --nnodes={number_of_nodes} --node_rank={rank_of_this_node}
        --nproc_per_node={number_of_gpus} {your_script.py} {your_arguments...}

    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.

    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
    """

    def __init__(
        self,
        task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        scheduler=None,
        gpus=None,
        batch_size=1,
        gradient_interval=1,
        num_worker=0,
        logger="logging",
        log_interval=100,
    ):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += (
                        ". Did you launch with `python -m torch.distributed.launch`?"
                    )
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params) :]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(
            log_interval=log_interval, silent=self.rank > 0, logger=logger
        )
        self.meter.log_config(self.config_dict())

    def train(self, num_epoch=1, batch_per_epoch=None):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(
            self.train_set, self.world_size, self.rank
        )
        dataloader = data.DataLoader(
            self.train_set,
            self.batch_size,
            sampler=sampler,
            num_workers=self.num_worker,
        )
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device], find_unused_parameters=True
                )
            else:
                model = nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True
                )
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss doesn't require grad. Did you define any loss in the task?"
                    )
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(
                        batch_per_epoch - start_id, self.gradient_interval
                    )

            if self.scheduler:
                try:
                    self.scheduler.step(loss)
                except IndexError:
                    pass

        return metric

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(
            test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker
        )
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            pred, target = model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        return metric, preds, targets

    def load(self, checkpoint, load_optimizer=True, strict=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"], strict=strict)

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, checkpoint)

        comm.synchronize()

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError(
                "Expect config class to be `%s`, but found `%s`"
                % (cls.__name__, config["class"])
            )

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id


@torch.no_grad()
def ema(ema_model, model, decay):
    msd = model.state_dict()
    for k, ema_v in ema_model.state_dict().items():
        model_v = msd[k].detach()
        ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


@R.register("core.EngineCV")
class EngineCV(core.Configurable):
    def __init__(
        self,
        task,
        dataset=None,
        optimizer=None,
        scheduler=None,
        gpus=None,
        n_folds=5,
        batch_size=1,
        gradient_interval=1,
        num_worker=0,
        logger="logging",
        log_interval=100,
    ):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.n_folds = n_folds

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += (
                        ". Did you launch with `python -m torch.distributed.launch`?"
                    )
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(dataset)
            if result is not None:
                dataset = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params) :]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.dataset = dataset

        self.dataset_splits = torch.utils.data.random_split(
            self.dataset,
            [len(self.dataset) // self.n_folds] * (self.n_folds - 1)
            + [
                len(self.dataset)
                - len(self.dataset) // self.n_folds * (self.n_folds - 1)
            ],
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(
            log_interval=log_interval, silent=self.rank > 0, logger=logger
        )
        self.meter.log_config(self.config_dict())

    def reset_model_and_epoch(self) -> None:
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.model.apply(fn=weight_reset)
        self.meter.epoch_id = 0

    def train(self, train_loader, train_sampler, num_epoch=1, batch_per_epoch=None):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            train_set (data.Dataset): training set
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """

        batch_per_epoch = batch_per_epoch or len(train_loader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device], find_unused_parameters=True
                )
            else:
                model = nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True
                )
        model.train()

        for epoch in self.meter(num_epoch):
            train_sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            for batch_id, batch in enumerate(islice(train_loader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = model(batch)
                if not loss.requires_grad:
                    raise RuntimeError(
                        "Loss doesn't require grad. Did you define any loss in the task?"
                    )
                loss = loss / gradient_interval
                loss.backward()
                metrics.append(metric)

                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(
                        batch_per_epoch - start_id, self.gradient_interval
                    )

            if self.scheduler:
                try:
                    self.scheduler.step(loss)
                except IndexError:
                    pass

    @torch.no_grad()
    def evaluate(self, val_loader, log=True):
        """
        Evaluate the model.

        Parameters:
            val_dataset (data.Dataset): validation set
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        model = self.model

        model.eval()
        preds = []
        targets = []
        for batch in val_loader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            pred, target = model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch")

        return metric, pred, target

    def one_train_val_loop(
        self,
        train_dataset,
        val_dataset,
        num_epoch: int = 10,
        val_interval: int = 3,
        early_stop: int = 5,
        weight_target: Optional[Union[List[float], torch.Tensor]] = None,
        ema_decay: float = 0.99,
    ):
        train_sampler = torch_data.DistributedSampler(
            train_dataset, self.world_size, self.rank
        )
        train_loader = data.DataLoader(
            train_dataset,
            self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_worker,
        )

        val_sampler = torch_data.DistributedSampler(
            val_dataset, self.world_size, self.rank
        )
        val_loader = data.DataLoader(
            val_dataset,
            self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_worker,
        )
        assert num_epoch > val_interval
        "Number of epochs must be greater than validation interval"
        n_loop = num_epoch // val_interval

        best_val_loss = 1e9
        no_improvement = 0

        ema_model = copy.deepcopy(self.model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

        best_model = copy.deepcopy(ema_model)
        best_pred = None
        for loop in range(n_loop):
            self.train(train_loader, train_sampler, num_epoch=val_interval)
            ema(ema_model, self.model, ema_decay)
            val_loss, pred, target = self.evaluate(val_loader, log=True)
            if weight_target is None:
                val_loss_list = torch.tensor(
                    list(val_loss.values()), dtype=torch.float32
                )
                val_loss_mean = torch.mean(val_loss_list)
            else:
                val_loss_list = torch.tensor(
                    list(val_loss.values()), dtype=torch.float32
                )
                val_loss_mean = torch.sum(val_loss_list * weight_target) / torch.sum(
                    weight_target
                )
            logger.warning(
                "Epoch [{}/{}], Val Loss: {:.4f}".format(
                    (loop + 1) * val_interval, num_epoch, val_loss_mean
                )
            )
            if val_loss_mean > best_val_loss:
                no_improvement += 1
                if no_improvement == early_stop:
                    logger.warning("Early stopping due to no improvement.")
                    break
            else:
                no_improvement = 0
                best_val_loss = val_loss_mean
                best_pred = pred
                best_model = copy.deepcopy(ema_model)

        return best_model, best_pred, target, best_val_loss

    def k_fold_train_val(
        self, num_epoch=1, val_interval=3, early_stop=15, ema_decay=0.99
    ):
        val_losses = torch.zeros(self.n_folds)
        y_preds = torch.tensor([])
        y_trues = torch.tensor([])
        models = []
        for fold in range(self.n_folds):
            logger.warning("Fold:", fold)
            torch.manual_seed(0)

            val_dataset = self.dataset_splits[fold]
            train_dataset = torch.utils.data.ConcatDataset(
                self.dataset_splits[:fold] + self.dataset_splits[fold + 1 :]
            )
            # TODO target is not right
            best_model, best_pred, target, val_loss = self.one_train_val_loop(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_epoch=num_epoch,
                val_interval=val_interval,
                early_stop=early_stop,
                weight_target=None,
                ema_decay=ema_decay,
            )
            models.append(best_model)
            y_preds = torch.cat((y_preds, best_pred.clone().detach()))
            y_trues = torch.cat((y_trues, target.clone().detach()))
            val_losses[fold] = val_loss
            logger.warning("\nBest val loss of {} fold: {:.4f}".format(fold, val_loss))
            self.reset_model_and_epoch()
        best_idx = torch.argmin(val_losses).item()
        return models[best_idx], y_preds, y_trues

    def load(self, checkpoint, load_optimizer=True, strict=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"], strict=strict)

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            torch.save(state, checkpoint)

        comm.synchronize()

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError(
                "Expect config class to be `%s`, but found `%s`"
                % (cls.__name__, config["class"])
            )

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
