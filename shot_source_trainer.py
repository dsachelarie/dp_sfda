import sys
import numpy as np
import torch
import wandb
import warnings
from torch import nn, optim
from nets.custom_wide_resnet import WideResNet
from torchvision import models
from torchvision import transforms
from custom_dataset import CustomDataset
from nets.shot_model import ShotModel
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from shot_config import ShotConfig
from utils import get_accuracy, CrossEntropyLabelSmooth

warnings.simplefilter("ignore")


class ShotSourceTrainer:
    def __init__(self, config: ShotConfig, trial: int):
        self.epochs = config.epochs
        self.device = config.device
        self.criterion = CrossEntropyLabelSmooth(config.no_classes, 0.1)
        self.backbone = config.backbone
        self.e = config.e
        self.domain = config.source_domain
        self.trial = trial

        if self.e != -1:
            self.privacy_engine = PrivacyEngine(secure_mode=False)
            self.max_physical_batch_size = config.max_physical_batch_size

        train_data = CustomDataset(config.data_path, config.transform, 1)

        model = ShotModel(config.backbone, config.no_classes)
        
        self.model = model.to(self.device) if self.e == -1 or ModuleValidator.is_valid(model) else ModuleValidator.fix(model).to(self.device)

        if config.pretrained:
            param_group = []
            for k, v in self.model.backbone.named_parameters():
                param_group += [{'params': v, 'lr': config.lr*0.1}]
            for k, v in self.model.bottleneck.named_parameters():
                param_group += [{'params': v, 'lr': config.lr}]
            for k, v in self.model.classifier.named_parameters():
                param_group += [{'params': v, 'lr': config.lr}]  

            self.optimizer = optim.SGD(param_group)

        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr) # weight_decay=1e-4 momentum=0.9

        # if self.e == -1:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr0'] = param_group['lr']

        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size)

        if self.e != -1:
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                epochs=config.epochs,
                target_epsilon=self.e,
                target_delta=config.d,
                max_grad_norm=config.c
            )

        wandb.login(key="38e8a7ccc52b05f2b5be452f7b1edb5631279dfb")
        wandb.init(
            project="shot_source",
            # project="dp_test",
            config={
                "dataset": config.dataset,
                "domain": config.source_domain,
                "epochs": config.epochs,
                "sigma": self.optimizer.noise_multiplier if config.e != -1 else None,
                "c": config.c if config.e != -1 else None,
                "epsilon": self.e,
                "delta": config.d if config.e != -1 else None,
                "learning_rate": config.lr,
                "batch_size": config.batch_size
            }
        )

    def lr_scheduler(self, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True

    def train_step(self, images, target, losses, top1_acc):
        self.optimizer.zero_grad()
        images, target = images.to(self.device), target.to(self.device)

        # if self.e == -1:
        #     self.lr_scheduler(iter_num=i + epoch * len(self.train_loader) + 1, max_iter=self.epochs * len(self.train_loader))

        output = self.model(images)
        loss = self.criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc = get_accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

        loss.backward()
        self.optimizer.step()

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            losses = []
            top1_acc = []

            # if self.e == -1:
            #     self.lr_scheduler(iter_num=epoch, max_iter=self.epochs)

            if self.e != -1:
                with BatchMemoryManager(
                        data_loader=self.train_loader,
                        max_physical_batch_size=self.max_physical_batch_size,
                        optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    for images, target, _ in memory_safe_data_loader:
                        self.train_step(images, target, losses, top1_acc)

            else:
                for images, target, _ in self.train_loader:
                    self.train_step(images, target, losses, top1_acc)

            wandb.log({"accuracy": np.mean(top1_acc), "loss": np.mean(losses)})
            
        torch.save(self.model.state_dict(), f"weights/{self.backbone}_{int(self.e)}_{self.domain}_{self.trial}.pth")
        wandb.finish()

    def test(self):
        self.model.eval()

        losses = 0
        top1_acc = 0

        with torch.no_grad():
            for images, target in self.test_loader:
                images = images.to(self.device)
                target = target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = get_accuracy(preds, labels)

                losses += loss.item()
                top1_acc += acc

        wandb.run.summary.update({"test_accuracy": top1_acc / len(self.test_loader),
                                  "test_loss": losses / len(self.test_loader)})
        wandb.finish()
