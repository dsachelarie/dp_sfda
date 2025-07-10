import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from collections import OrderedDict
from custom_dataset import CustomDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from nets.shot_model import ShotModel
from sklearn.metrics import roc_curve, auc
from torch import optim
from torchvision import transforms
from utils import CrossEntropyLabelSmooth, get_accuracy

transform = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = CustomDataset("./data/small_objects/cifar10/train", transform, 1)
out_of_distribution_data = CustomDataset("./data/small_objects/stl10/train", transform, 1, out_of_distribution=True)
criterion = CrossEntropyLabelSmooth(9, 0.1)

# loader = torch.utils.data.DataLoader(data, batch_size=128)

plt.figure(figsize=(6, 6))

colors = {
    "8": "red",
    "4": "green",
    "1": "orange",
    "0.1": "purple"
}

for epsilon in ["8"]: # ["-1", "8", "4", "1", "0.1"]:
    privacy_engine = PrivacyEngine(secure_mode=False)
    source_data_split = data.sample()
    source_data_ids = source_data_split.original_indices
    target_data_split = data.sample()
    target_data_split.join(out_of_distribution_data)
    target_data_ids = target_data_split.original_indices

    if epsilon == "-1":
        loader = torch.utils.data.DataLoader(source_data_split, batch_size=64)

        model = ShotModel("alexnet", 9)
        # state_dict = torch.load("weights/lenet_-1_mnist_0.pth", map_location=torch.device(device))

        optimizer = optim.SGD(model.parameters(), 0.01)

    else:
        loader = torch.utils.data.DataLoader(source_data_split, batch_size=256)

        model = ShotModel("alexnet", 9)
        # state_dict = torch.load(f"weights/alexnet_{epsilon}_mnist_0.pth", map_location=torch.device(device))
        optimizer = optim.SGD(model.parameters(), 0.01)

        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            epochs=30,
            target_epsilon=int(epsilon), 
            target_delta=1e-5,
            max_grad_norm=3
        )

    model = model.to(device)
    model.train()

    for epoch in range(30):
        top1_acc = []

        if epsilon != "-1":
            with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=64,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                for images, target, _ in memory_safe_data_loader:
                    optimizer.zero_grad()
                    images, target = images.to(device), target.to(device)

                    output = model(images)
                    loss = criterion(output, target)

                    preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                    labels = target.detach().cpu().numpy()
                    acc = get_accuracy(preds, labels)
                    top1_acc.append(acc)

                    loss.backward()
                    optimizer.step()

        else:
            for images, target, _ in loader:
                optimizer.zero_grad()
                images, target = images.to(device), target.to(device)

                output = model(images)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()
                acc = get_accuracy(preds, labels)
                top1_acc.append(acc)

                loss.backward()
                optimizer.step()

        print(np.mean(top1_acc), file=sys.stderr)

    target_data_split.labels = [1 if x in source_data_ids else 0 for x in target_data_ids]
    loader = torch.utils.data.DataLoader(target_data_split, batch_size=128)

    # new_state_dict = OrderedDict()
    # for key, value in state_dict.items():
    #     new_key = key.replace("_module.", "")
    #     new_state_dict[new_key] = value

    # model.load_state_dict(new_state_dict)
    # model = model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []

    for i, (images, target, _) in enumerate(loader):
        images, target = images.to(device), target.to(device)

        with torch.no_grad():
            output = model(images)

            scores = torch.max(torch.softmax(output, dim=1), dim=1).values
            all_outputs.extend(scores.cpu().tolist())
            all_labels.extend(target.cpu().tolist())

        print(f"{i}/{len(loader)}", file=sys.stderr)

    print(all_outputs[:100], file=sys.stderr)
    print(all_labels[:100], file=sys.stderr)

    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)

    if epsilon == "-1":
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"Non-private source classifier (AUC = {roc_auc:.2f})")

    else:
        plt.plot(fpr, tpr, color=colors[epsilon], lw=2, label=f"Private source classifier, ε={epsilon} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label="Random guess")
# plt.xscale('log')
# plt.yscale('log')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.axis("square")
plt.xlim([1e-5, 1.0])
plt.ylim([1e-5, 1.0])
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()
