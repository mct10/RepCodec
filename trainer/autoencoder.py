# Copyright (c) ByteDance, Inc. and its affiliates.
# Copyright (c) Chutong Meng
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.
# Based on AudioDec (https://github.com/facebookresearch/AudioDec)

import logging
import os
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger("repcodec_train")


class Trainer:
    def __init__(
            self,
            steps: int,
            epochs: int,
            data_loader: dict,
            model: dict,
            criterion: dict,
            optimizer: dict,
            scheduler: dict,
            config: dict,
            device=torch.device("cpu"),
    ):
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.train_max_steps = config.get("train_max_steps", 0)

    def _train_step(self, batch):
        """Single step of training."""
        mode = "train"
        x = batch
        x = x.to(self.device)

        codec_loss = 0.0
        y_, zq, z, vqloss, perplexity = self.model["repcodec"](x)
        self._perplexity(perplexity, mode=mode)
        codec_loss += self._vq_loss(vqloss, mode=mode)
        codec_loss += self._metric_loss(y_, x, mode=mode)

        self._record_loss("codec_loss", codec_loss, mode=mode)
        self._update_repcodec(codec_loss)

        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Single step of evaluation."""
        mode = "eval"
        x = batch
        x = x.to(self.device)

        codec_loss = 0.0
        y_, zq, z, vqloss, perplexity = self.model["repcodec"](x)
        self._perplexity(perplexity, mode=mode)
        codec_loss += self._vq_loss(vqloss, mode=mode)
        codec_loss += self._metric_loss(y_, x, mode=mode)

        self._record_loss("codec_loss", codec_loss, mode=mode)

    def run(self):
        """Run training."""
        self.finish_train = False
        self.tqdm = tqdm(
            initial=self.steps, total=self.train_max_steps, desc="[train]"
        )
        while True:
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path: str):
        state_dict = {
            "model": {
                "repcodec": self.model["repcodec"].state_dict()
            },
            "optimizer": {
                "repcodec": self.optimizer["repcodec"].state_dict(),
            },
            "scheduler": {
                "repcodec": self.scheduler["repcodec"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(
            self,
            checkpoint_path: str,
            strict: bool = True,
            load_only_params: bool = False
    ):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["repcodec"].load_state_dict(
            state_dict["model"]["repcodec"], strict=strict
        )

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["repcodec"].load_state_dict(
                state_dict["optimizer"]["repcodec"]
            )
            self.scheduler["repcodec"].load_state_dict(
                state_dict["scheduler"]["repcodec"]
            )

    def _train_epoch(self):
        """One epoch of training."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        if train_steps_per_epoch > 200:
            logger.info(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({self.train_steps_per_epoch} steps per epoch)."
            )

    def _eval_epoch(self):
        """One epoch of evaluation."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
                tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    def _metric_loss(self, predict_y, natural_y, mode='train'):
        """Metric losses."""
        metric_loss = 0.0

        repr_reconstruct_loss = self.criterion["repr_reconstruct_loss"](predict_y, natural_y)
        repr_reconstruct_loss *= self.config["lambda_repr_reconstruct_loss"]
        self._record_loss("reconstruct_loss", repr_reconstruct_loss, mode=mode)
        metric_loss += repr_reconstruct_loss

        return metric_loss

    def _update_repcodec(self, repr_loss):
        """Update generator."""
        self.optimizer["repcodec"].zero_grad()
        repr_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["repcodec"].parameters(),
                self.config["grad_norm"],
            )
        self.optimizer["repcodec"].step()
        self.scheduler["repcodec"].step()

    def _record_loss(self, name: str, loss, mode='train'):
        """Record loss."""
        if torch.is_tensor(loss):
            loss = loss.item()

        if mode == 'train':
            self.total_train_loss[f"train/{name}"] += loss
        elif mode == 'eval':
            self.total_eval_loss[f"eval/{name}"] += loss
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps and (self.steps % self.config["save_interval_steps"] == 0):
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.train_max_steps:
            self.finish_train = True
        else:
            self.finish_train = False
        return self.finish_train

    def _perplexity(self, perplexity, label=None, mode='train'):
        if label:
            name = f"{mode}/ppl_{label}"
        else:
            name = f"{mode}/ppl"
        if torch.numel(perplexity) > 1:
            perplexity = perplexity.tolist()
            for idx, ppl in enumerate(perplexity):
                self._record_loss(f"{name}_{idx}", ppl, mode=mode)
        else:
            self._record_loss(name, perplexity, mode=mode)

    def _vq_loss(self, vqloss, label=None, mode='train'):
        if label:
            name = f"{mode}/vqloss_{label}"
        else:
            name = f"{mode}/vqloss"
        vqloss = torch.sum(vqloss)
        vqloss *= self.config["lambda_vq_loss"]
        self._record_loss(name, vqloss, mode=mode)

        return vqloss
