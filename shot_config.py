import torch
import argparse
from torchvision import transforms


class ShotConfig:
    def __init__(self, args: argparse.Namespace, target: bool, trial=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.dataset == "mnist":
            self._setup_digits(args, target, trial)

        elif args.dataset == "office-31":
            self._setup_office31(args, target, trial)

        elif args.dataset == "small_objects":
            self._setup_small_objects(args, target, trial)

        else:
            raise argparse.ArgumentError(None, "Invalid dataset")

    def _common_target_initialization(self, trial):
        state_dict = torch.load(f"weights/{self.backbone}_{self.source_e}_{self.source_domain}_{trial}.pth", map_location=self.device)
        # state_dict = torch.load(f"weights/{self.backbone}_{self.source_e}_generated.pth", map_location=self.device)
        # state_dict = torch.load(f"weights/alexnet_8_mnist.pth", map_location=self.device)

        # Remove "_module." prefix from keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key.replace("_module.", "")
            new_state_dict[new_key] = value

        self.weights = new_state_dict

    def _setup_digits(self, args: argparse.Namespace, target: bool, trial: int):
        self.no_classes = 10
        self.dataset = "mnist"
        self.source_domain = args.source_domain
        self.e = args.e
        self.source_e = args.source_e if target else None
        self.target_domain = args.target_domain if target else None
        self.beta = 0.1

        if args.backbone in ["alexnet", "lenet", "dtn"]:
            self.backbone = args.backbone
            # self.backbone = "lenet"
            self.pretrained = False
            self.transform = transforms.Compose([
                transforms.CenterCrop((28, 28)),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4,
                #     hue=0.1 
                # ),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            if target:
                self.test_transform = transforms.Compose([
                    transforms.CenterCrop((28, 28)),
                    transforms.ToTensor()
                ])
                self.train_data_path = f"./data/{self.dataset}/{self.target_domain}/train"
                self.test_data_path = f"./data/{self.dataset}/{self.target_domain}/test"
                self._common_target_initialization(trial)
                self.few_shot = args.use_few_shot

                if self.e != -1:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 64 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/mnist/{self.source_domain}/train"
                # self.data_path = f"./generated"

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 0.1 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 512 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs

        elif args.backbone == "wideresnet":
            self.backbone = "wideresnet"
            self.pretrained = False
            self.transform = transforms.Compose([
                transforms.CenterCrop((28, 28)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
            if target:
                self.data_path = f"./data/{self.dataset}/{self.target_domain}/train"
                self._common_target_initialization(trial)

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 512 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/mnist/{self.source_domain}/train"

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 1 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 512 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 1024 if args.batch_size is None else args.batch_size
                    self.lr = 0.001 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs

        elif args.backbone == "dirt-t":
            self.backbone = "dirt-t_cnn"
            self.pretrained = False
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
        
            if target:
                self.data_path = f"./data/{self.dataset}/{self.target_domain}/train"
                self._common_target_initialization(trial)

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 1 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 1024 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    self.lr = 0.1 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/mnist/{self.source_domain}/train"

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 0.1 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 1024 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.002 if args.lr is None else args.lr
                    self.epochs = 100 if args.epochs is None else args.epochs

    def _setup_small_objects(self, args: argparse.Namespace, target: bool, trial: int):
        self.no_classes = 9
        self.dataset = "small_objects"
        self.source_domain = args.source_domain
        self.target_domain = args.target_domain if target else None
        self.e = args.e
        self.source_e = args.source_e if target else None
        self.beta = 0.1

        if args.backbone == "wideresnet":
            self.backbone = "wideresnet"
            self.pretrained = False
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # These values, specific to the CIFAR10 dataset, are assumed to be known.
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            if target:
                self.data_path = f"./data/{self.dataset}/{self.target_domain}/train"
                self._common_target_initialization(trial)

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 1024 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    self.lr = 0.001 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/small_objects/{self.source_domain}/train"

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 3 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 256 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 1 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs

        elif args.backbone == "resnet50":
            self.backbone = "resnet50"
            self.pretrained = False
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                # These values, specific to the CIFAR10 dataset, are assumed to be known.
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            if target:
                self.data_path = f"./data/{self.dataset}/{self.target_domain}/train"
                self._common_target_initialization(trial)

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 1024 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    self.lr = 0.001 if args.lr is None else args.lr
                    self.epochs = 15 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/small_objects/{self.source_domain}/train"

                if self.e != -1:
                    self.batch_size = 4096 if args.batch_size is None else args.batch_size
                    self.lr = 3 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 256 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 1 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs

    def _setup_office31(self, args: argparse.Namespace, target: bool, trial: int):
        self.no_classes = 31
        self.dataset = "office-31"
        self.source_domain = args.source_domain
        self.target_domain = args.target_domain if target else None
        self.e = args.e
        self.source_e = args.source_e if target else None
        self.beta = 0.3

        if args.backbone == "resnet50":
            self.backbone = "resnet50"
            self.pretrained = True
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            if target:
                self.train_data_path = f"./data/{self.dataset}/{self.target_domain}/images"
                self.test_data_path = f"./data/{self.dataset}/{self.target_domain}/images"
                self._common_target_initialization(trial)
                self.test_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
                self.few_shot = args.use_few_shot

                if self.e != -1:
                    # TODO change
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    # self.lr = 0.1 if args.lr is None else args.lr
                    self.lr = 0.01 if args.lr is None else args.lr # check if good
                    self.epochs = 15 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 64 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs

            else:
                self.data_path = f"./data/{self.dataset}/{self.source_domain}/images"

                if self.e != -1:
                    # TODO change
                    self.batch_size = 128 if args.batch_size is None else args.batch_size
                    self.lr = 0.1 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
                    self.max_physical_batch_size = 64 if args.max_physical_batch_size is None else args.max_physical_batch_size
                    self.c = 3 if args.c is None else args.c
                    self.d = 1e-5 if args.d is None else args.d

                else:
                    self.batch_size = 64 if args.batch_size is None else args.batch_size
                    self.lr = 0.01 if args.lr is None else args.lr
                    self.epochs = 30 if args.epochs is None else args.epochs
