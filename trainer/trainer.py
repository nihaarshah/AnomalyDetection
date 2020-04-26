# Code from github.com/victoresque/pytorch-template/
import numpy as np
import torch

from model.evaluation import top_n_percent_anomaly

# from torchvision.utils import make_grid
from utils import MetricTracker, inf_loop

from .base_trainer import BaseTrainer
from .kl_schedule import kl_scheduler, start_stop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        data_loader,
        valid_data_loader=None,
        test_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.FloatTensor = (
            torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        )
        self.total_iter = self.len_epoch * self.epochs

        # KL annealing strategy
        self.kl_strategy = config.config["kl_strategy"]
        self.kl_loss = self.kl_strategy["loss"]
        self.kl_schedule = self.kl_strategy["schedule"]
        start, stop = start_stop(
            self.kl_strategy["beta_start"],
            self.kl_strategy["beta_stop"],
            self.kl_strategy["C_start"],
            self.kl_strategy["C_stop"],
            self.kl_loss,
        )

        self.kl_weight_param_schedule = kl_scheduler(
            n_iter=self.total_iter,
            start=start,
            stop=stop,
            n_cycle=self.kl_strategy["n_cycle"],
            ratio=self.kl_strategy["ratio"],
            schedule=self.kl_schedule,
        )
        self.gamma = self.kl_strategy["gamma"]
        self.recon_loss = self.kl_strategy["recon_loss"]

        if self.config["data_loader"]["args"]["scaling"] == "binary":
            assert self.recon_loss == "bce"

        # Metrics
        self.train_metrics = MetricTracker(
            "loss", "recon", "kld", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "recon", "kld", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            step = (epoch - 1) * self.len_epoch + batch_idx

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, z_mu, z_var, ldj, z_0, z_k = self.model(data)
            loss, recon, kld, kl_weight_param = self.criterion(
                output=output,
                target=target,
                z_mu=z_mu,
                z_var=z_var,
                z_0=z_0,
                z_k=z_k,
                ldj=ldj,
                kl_weight_param=self.kl_weight_param_schedule[step],
                gamma=self.gamma,
                recon_loss=self.recon_loss,
                kl_loss=self.kl_loss,
            )
            loss.backward()
            self.optimizer.step()

            self.writer.set_step(step)
            self.train_metrics.update("loss", loss.item())
            self.train_metrics.update("recon", recon.item())
            self.train_metrics.update("kld", kld.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.2f}, Recon: {:.6f}, KLD: {:.2f}, KL weight param: {:.2f}".format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        recon.item(),
                        kld.item(),
                        kl_weight_param.item(),
                    )
                )
                # self.writer.add_image('input', make_grid(
                #     data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, z_mu, z_var, ldj, z_0, z_k = self.model(data)
                mse = torch.nn.MSELoss()
                loss, recon, kld, kl_weight_param = self.criterion(
                    output=output,
                    target=target,
                    z_mu=z_mu,
                    z_var=z_var,
                    z_0=z_0,
                    z_k=z_k,
                    ldj=ldj,
                    kl_weight_param=1,
                    gamma=self.gamma,
                    recon_loss=self.recon_loss,
                    kl_loss="weight",
                )
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid",
                )
                self.valid_metrics.update("loss", loss.item())
                self.valid_metrics.update("recon", recon.item())
                self.valid_metrics.update("kld", kld.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(
                #     data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def evaluate(self):

        self.model.eval()
        recon_losses = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # # p(x | mu, sigma)
                # output, z_mu, z_var, ldj, z_0, z_k = self.model(data)
                # loss, recon, kld, kl_weight_param = self.criterion(
                #     output=output,
                #     target=data,
                #     z_mu=z_mu,
                #     z_var=z_var,
                #     z_0=z_0,
                #     z_k=z_k,
                #     ldj=ldj,
                #     kl_weight_param=1,
                #     gamma=self.gamma,
                #     recon_loss=self.recon_loss,
                #     kl_loss=self.kl_loss,
                #     pointwise=True,
                # )
                # recon_losses.extend(recon.numpy())
                # targets.extend(target.numpy())

                # E [ p(x | mu, sigma)]
                samples = 100
                # n*samples x cols
                inflated_data = data.repeat_interleave(samples, dim=0)
                output, z_mu, z_var, ldj, z_0, z_k = self.model(inflated_data)
                loss, recon, kld, kl_weight_param = self.criterion(
                    output=output,
                    target=inflated_data,
                    z_mu=z_mu,
                    z_var=z_var,
                    z_0=z_0,
                    z_k=z_k,
                    ldj=ldj,
                    kl_weight_param=1,
                    gamma=self.gamma,
                    recon_loss=self.recon_loss,
                    kl_loss=self.kl_loss,
                    pointwise=True,
                )
                recon_losses.extend(recon.reshape(data.shape[0], -1).mean(axis=1).cpu().numpy())
                targets.extend(target.cpu().numpy())

        import pickle

        pickle.dump((recon_losses, targets), open("temp_results.pickle", "wb"))

        anomaly_metrics = top_n_percent_anomaly(
            recon_losses, targets, dataset=self.config["data_loader"]["args"]["dataset"]
        )
        self.logger.info("================ Test Results ===================")
        for key, value in anomaly_metrics.items():
            self.logger.info("    {:15s}: {}".format(str(key), value))

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
