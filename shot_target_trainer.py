import sys
import numpy as np
import torch
import wandb
import warnings
import copy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from nets.custom_wide_resnet import WideResNet
from nets.generator import Generator
from nets.discriminator import Discriminator
from nets.da_discriminator import DADiscriminator
from torchvision import models
from torchvision import transforms
from custom_dataset import CustomDataset
from nets.shot_model import ShotModel
from opacus.validators import ModuleValidator
from scipy.spatial.distance import cdist
from shot_config import ShotConfig
from PIL import Image
from vat import VAT

warnings.simplefilter("ignore")


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy 


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get_accuracy(preds, labels):
    return np.mean(preds == labels)


class ShotTargetTrainer:
    def __init__(self, config: ShotConfig):
        self.e = config.e
        self.epochs = config.epochs
        self.device = config.device
        self.criterion = nn.CrossEntropyLoss()
        self.beta = config.beta
        self.lr = config.lr
        self.no_classes = config.no_classes
        self.few_shot = config.few_shot

        if self.e != -1:
            self.privacy_engine = PrivacyEngine(secure_mode=False)
            self.max_physical_batch_size = config.max_physical_batch_size

        if config.train_data_path == config.test_data_path:
            train_perc = 0.9
            test_perc = 0.1

        else:
            train_perc = 1
            test_perc = 1

        train_data = CustomDataset(config.train_data_path, config.transform, train_perc)
        pseudo_label_data = CustomDataset(config.train_data_path, config.transform, train_perc, test=True)
        test_data = CustomDataset(config.test_data_path, config.test_transform, test_perc, test=True)
        few_shot_data = CustomDataset(config.train_data_path, config.transform, train_perc, few_shot=True)

        model = ShotModel(config.backbone, self.no_classes)

        model.load_state_dict(config.weights)

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
            self.optimizer = optim.SGD(model.parameters(), lr=config.lr)

        # for param_group in self.optimizer.param_groups:
        #     param_group['lr0'] = param_group['lr']

        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size) # shuffle doesn't work when physical batch size is smaller than actual batch size
        self.pseudo_label_loader = torch.utils.data.DataLoader(pseudo_label_data, batch_size=config.max_physical_batch_size if self.e != -1 else config.batch_size)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.max_physical_batch_size if self.e != -1 else config.batch_size)
        self.few_shot_loader = torch.utils.data.DataLoader(few_shot_data, batch_size=3 * 10)

        for param in self.model.classifier.parameters():
            param.requires_grad = False

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

        self.local_rank = 0

        if self.local_rank == 0:
            wandb.login(key="38e8a7ccc52b05f2b5be452f7b1edb5631279dfb")
            wandb.init(
                project="shot_target",
                config={
                    "dataset": config.dataset,
                    "source_domain": config.source_domain,
                    "target_domain": config.target_domain,
                    "epochs": self.epochs,
                    "sigma": self.optimizer.noise_multiplier if config.e != -1 else None,
                    "c": config.c if config.e != -1 else None,
                    "epsilon": self.e,
                    "source_epsilon": config.source_e,
                    "delta": config.d if config.e != -1 else None,
                    "learning_rate": config.lr,
                    "batch_size": config.batch_size
                }
            )

        # Below generator parameters
        self.generator = Generator(100).to(self.device)
        # self.generator = DDP(self.generator, device_ids=[self.local_rank])
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator = Discriminator().to(self.device)
        # self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank])
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.generator_criterion = nn.BCELoss()

        self.da_discriminator = DADiscriminator().to(self.device)
        self.da_discriminator_optimizer = optim.Adam(self.da_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # if self.e != -1:
        #     self.discriminator, self.discriminator_optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
        #         module=self.discriminator,
        #         optimizer=self.discriminator_optimizer,
        #         data_loader=self.train_loader,
        #         epochs=config.epochs,
        #         target_epsilon=self.e,
        #         target_delta=config.d,
        #         max_grad_norm=config.c
        #     )

        if torch.cuda.device_count() > 1:
            self.generator = DistributedDataParallel(self.generator)
            self.discriminator = DistributedDataParallel(self.discriminator)

    def lr_scheduler(self, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True

    def get_pseudo_labels(self, epoch):
        self.model.eval()

        start_test = True
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(self.pseudo_label_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                feas = self.model.bottleneck(torch.flatten(self.model.backbone(images), 1))
                outputs = self.model.classifier(feas)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False

                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
        unknown_weight = 1 - ent / np.log(self.no_classes)
        _, predict = torch.max(all_output, 1)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()

        i = 0

        if self.few_shot:
            # Get few-shot inputs and labels
            few_images, few_labels, _ = next(iter(self.few_shot_loader))
            few_images = few_images.to(self.device)
            few_labels = few_labels.to(self.device)

            # Extract features (same way as rest of data)
            with torch.no_grad():
                few_fea = self.model.bottleneck(torch.flatten(self.model.backbone(few_images), 1))

            # Normalize features the same way
            few_fea = torch.cat((few_fea, torch.ones(few_fea.size(0), 1).to(few_fea.device)), 1)
            few_fea = (few_fea.t() / torch.norm(few_fea, p=2, dim=1)).t()
            few_fea = few_fea.float().cpu().numpy()
            few_labels = few_labels.cpu().numpy()

            # Convert labels to one-hot
            few_aff = np.eye(K)[few_labels.astype(int)]

            initc = few_aff.transpose().dot(few_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], "cosine")
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

            i += 1

            # Add to main arrays
            # all_fea = few_fea
            # aff = few_aff
            # all_label = torch.from_numpy(few_labels).float()
            # all_fea = np.concatenate((all_fea, few_fea), axis=0)
            # aff = np.concatenate((aff, few_aff), axis=0)
            # all_label = torch.cat((all_label, torch.from_numpy(few_labels).float()), dim=0)

        while i < 2:
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count>0)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], "cosine")
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]
            
            i += 1

        acc = np.sum(predict == all_label.cpu().float().numpy()) / len(all_fea)
        wandb.log({"pseudo-label accuracy": acc}, step=epoch)

        return predict.astype('int')

    def train_step(self, images, target, indices, pseudo_labels, losses, top1_acc):
        self.optimizer.zero_grad()
        images = images.to(self.device)
        target = target.to(self.device)

        # SHOT only
        # self.lr_scheduler(iter_num=i + epoch * len(self.train_loader) + 1, max_iter=self.epochs * len(self.train_loader))

        output = self.model(images)

        # print(pseudo_labels.shape, file=sys.stderr)

        loss = self.beta * self.criterion(output, pseudo_labels[indices])

        output_softmax = nn.Softmax(dim=1)(output)
        entropy_loss = torch.mean(Entropy(output_softmax))
        output_softmax_mean = output_softmax.mean(dim=0)
        gentropy_loss = torch.sum(-output_softmax_mean * torch.log(output_softmax_mean + 1e-5))
        entropy_loss -= gentropy_loss

        loss += entropy_loss

        # for fs_images, fs_target, _ in self.few_shot_loader:
        #     fs_images, fs_target = fs_images.to(self.device), fs_target.to(self.device)

        #     fs_output = self.model(fs_images)
        #     fs_loss = self.criterion(fs_output, fs_target)

        #     fs_preds = np.argmax(fs_output.detach().cpu().numpy(), axis=1)
        #     fs_labels = fs_target.detach().cpu().numpy()
        #     acc = get_accuracy(fs_preds, fs_labels)

        #     loss += 0.1 * fs_loss
        #     # fs_losses.append(fs_loss.item())

        #     # wandb.log({"few-shot accuracy": acc})

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc = get_accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

        loss.backward()
        self.optimizer.step()

    def train_synthetic_samples_generator(self):
        self.discriminator.train()
        self.generator.train()
        self.model.eval()

        for epoch in range(self.epochs):
            disc_losses = 0
            gen_losses = 0

            # with BatchMemoryManager(
            #         data_loader=self.train_loader,
            #         max_physical_batch_size=self.max_physical_batch_size,
            #         optimizer=self.discriminator_optimizer
            # ) as memory_safe_data_loader:
            #     for images, class_labels, _ in memory_safe_data_loader:
            for images, class_labels, _ in self.train_loader:
                images = images.to(self.device)
                class_labels = nn.functional.one_hot(class_labels, num_classes=10).to(self.device)

                # Train discriminator
                noise = torch.randn(len(images), 90, device=self.device)
                noise = torch.cat((noise, class_labels), dim=1)
                fake_images = self.generator(noise).detach()

                real_labels = torch.ones(len(images), 1, device=self.device)
                fake_labels = torch.zeros(len(images), 1, device=self.device)

                self.discriminator_optimizer.zero_grad()
                real_loss = self.generator_criterion(self.discriminator(images), real_labels)
                fake_loss = self.generator_criterion(self.discriminator(fake_images), fake_labels)
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.discriminator_optimizer.step()

                disc_losses += disc_loss.item()

                # Train generator
                self.discriminator_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()

                noise = torch.randn(len(images), 90, device=self.device)
                noise = torch.cat((noise, class_labels), dim=1)
                fake_samples = self.generator(noise)

                # Get source_classifier predictions on fake images
                preds = self.model(fake_samples)
                class_loss = 1 - torch.max(class_labels * nn.functional.softmax(preds, dim=1), dim=1)[0].mean()
                gen_loss = self.generator_criterion(self.discriminator(fake_samples), real_labels)
                gen_loss += class_loss
                gen_loss.backward()
                self.generator_optimizer.step()

                gen_losses += gen_loss.item()

            mean_disc_loss = disc_losses / len(self.train_loader)
            mean_gen_loss = gen_losses / len(self.train_loader)

            if self.local_rank == 0:
                wandb.log({"disc loss": mean_disc_loss, "gen loss": mean_gen_loss})

            # if mean_gen_loss < 0.15:
            #     break

        torch.save(self.generator.state_dict(), f"weights/generator.pth")

    def test_synthetic_samples_generator(self):
        self.generator.eval()
        self.model.eval()

        print("Testing synthetic samples generator", file=sys.stderr)

        self.generator.load_state_dict(torch.load(f"weights/generator.pth", map_location=self.device))
        self.generator.eval()

        with torch.no_grad():
            for label in range(10):
                one_hot_label = nn.functional.one_hot(
                    torch.tensor(label, dtype=torch.long), num_classes=10).to(self.device)

                noise = torch.randn(5000, 90, device=self.device)
                noise = torch.cat((noise, one_hot_label.unsqueeze(dim=0).repeat(5000, 1)), dim=1)
                generated_data = self.generator(noise)

                with torch.no_grad():
                    outputs = self.model(generated_data)
                    preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    labels = np.full(5000, label)

                    print(f"label: {label} accuracy: {get_accuracy(preds, labels)}")

                for i in range(5000):
                    image_np = (generated_data[i].cpu() * 255).byte().numpy()  # Convert to uint8

                    # Transpose from (C, H, W) to (H, W, C) for PIL
                    image_np = np.transpose(image_np, (1, 2, 0))

                    # Convert to PIL image
                    image_pil = Image.fromarray(image_np)

                    # Save as PNG or JPG
                    image_pil.save(f"generated/{label}/{i}.png")

                

                # preds = np.argmax(self.source_classifier(generated_data).detach().cpu().numpy(), axis=1)
                # print(f"Accuracy for label {label}: {get_accuracy(preds, np.repeat(label, 10))}",
                #       file=sys.stderr)

    def train_feature_extractor(self):
        self.da_discriminator.train()
        self.generator.load_state_dict(torch.load(f"weights/generator.pth", map_location=self.device))
        self.generator.eval()

        self.synth_feature_extractor = self.model.backbone
        self.synth_feature_extractor.eval()
        self.target_feature_extractor = copy.deepcopy(self.synth_feature_extractor)
        self.feature_extractor_optimizer = optim.Adam(self.target_feature_extractor.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.target_feature_extractor.train()

        for epoch in range(self.epochs):
            extr_losses = 0
            disc_losses = 0

            for target_images, class_labels, _ in self.train_loader:
                target_images = target_images.to(self.device)

                class_labels = nn.functional.one_hot(class_labels, num_classes=10).to(self.device)

                # Generate synthetic images
                noise = torch.randn(len(target_images), 90, device=self.device)
                noise = torch.cat((noise, class_labels), dim=1)
                synth_images = self.generator(noise)

                # Train discriminator
                synth_features = self.synth_feature_extractor(synth_images)
                target_features = self.target_feature_extractor(target_images).detach()

                target_labels = torch.ones(len(target_images), 1, device=self.device)
                synth_labels = torch.zeros(len(target_images), 1, device=self.device)

                self.da_discriminator_optimizer.zero_grad()
                target_loss = self.generator_criterion(self.da_discriminator(target_features), target_labels)
                synth_loss = self.generator_criterion(self.da_discriminator(synth_features), synth_labels)
                disc_loss = target_loss + synth_loss
                disc_loss.backward()
                self.da_discriminator_optimizer.step()

                disc_losses += disc_loss.item()

                # Train target feature extractor
                self.da_discriminator_optimizer.zero_grad()
                self.target_feature_extractor.zero_grad()
                target_features = self.target_feature_extractor(target_images)
                target_feature_loss = self.generator_criterion(self.da_discriminator(target_features), synth_labels)
                target_feature_loss.backward()
                self.feature_extractor_optimizer.step()

                extr_losses += target_feature_loss.item()

            mean_disc_loss = disc_losses / len(self.train_loader)
            mean_extr_loss = extr_losses / len(self.train_loader)

            if torch.cuda.current_device() == 0:
                wandb.log({"disc loss": mean_disc_loss, "target feature extractor loss": mean_extr_loss})

            # if mean_extr_loss < 0.01:
            #     break

        torch.save(self.target_feature_extractor.state_dict(), f"weights/synth_feature_extractor.pth")

    def few_shot_finetuning(self):
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = True
        # for param in self.model.bottleneck.parameters():
        #     param.requires_grad = True
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True

        # optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        self.model.train()

        for epoch in range(5):
            losses = []
            top1_acc = []

            for images, target, _ in self.few_shot_loader:
                self.optimizer.zero_grad()
                images, target = images.to(self.device), target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = get_accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

                loss.backward()
                self.optimizer.step()

            # wandb.log({"few-shot accuracy": np.mean(top1_acc), "few-shot loss": np.mean(losses)})

        # for param in self.model.parameters():
        #     param.requires_grad = True
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = False

    def train(self):
        # self.few_shot_finetuning()

        for epoch in range(self.epochs):
            losses = []
            top1_acc = []

            # self.few_shot_finetuning()
            pseudo_labels = torch.from_numpy(self.get_pseudo_labels(epoch)).to(self.device)

            self.model.train()
            # self.model.classifier.eval()

            if self.few_shot and epoch == 5:
                self.beta = 1

            if self.e != -1:
                with BatchMemoryManager(
                        data_loader=self.train_loader,
                        max_physical_batch_size=self.max_physical_batch_size,
                        optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    for images, target, indices in memory_safe_data_loader:
                        self.train_step(images, target, indices, pseudo_labels, losses, top1_acc)

            else:
                for images, target, indices in self.train_loader:
                    self.train_step(images, target, indices, pseudo_labels, losses, top1_acc)

            wandb.log({"accuracy": np.mean(top1_acc), "loss": np.mean(losses)}, step=epoch)

    def test(self, msg="test"):
        self.model.eval()

        losses = 0
        top1_acc = 0

        with torch.no_grad():
            for images, target, _ in self.test_loader:
                images = images.to(self.device)
                target = target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = get_accuracy(preds, labels)

                losses += loss.item()
                top1_acc += acc

                # print(preds, file=sys.stderr)
                # print(labels, file=sys.stderr)

        wandb.run.summary.update({f"{msg} accuracy": top1_acc / len(self.test_loader),
                                  f"{msg} loss": losses / len(self.test_loader)})

        return top1_acc / len(self.test_loader)
        # wandb.finish()

    def test_feature_extractor(self):
        self.model.backbone.load_state_dict(torch.load(f"weights/synth_feature_extractor.pth", map_location=self.device))
        self.model.eval()

        losses = 0
        top1_acc = 0

        with torch.no_grad():
            for images, target, _ in self.test_loader:
                images = images.to(self.device)
                target = target.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = get_accuracy(preds, labels)

                losses += loss.item()
                top1_acc += acc

                # print(preds, file=sys.stderr)
                # print(labels, file=sys.stderr)

        wandb.run.summary.update({"test accuracy": top1_acc / len(self.test_loader),
                                  "test loss": losses / len(self.test_loader)})
        # wandb.finish()
