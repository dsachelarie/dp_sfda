import numpy as np
import os
import random
import sys
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, transform, perc, test=False, few_shot=False, lira=False, path2=None, out_of_distribution=False):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        self.class_labels = {class_name: i for i, class_name in enumerate(self.classes)}

        images = []
        for class_name in self.classes:
            class_folder = os.path.join(path, class_name)
            class_images = []

            for i, file_name in enumerate(os.listdir(class_folder)):
                if few_shot and i >= 3 or lira and i >= 2 * 100:
                    break

                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    images.append((os.path.join(class_folder, file_name),
                                    self.class_labels[class_name]))

            if not test and not few_shot:
                class_images = class_images[:int(len(class_images) * perc)]
                
            elif not few_shot:
                class_images = class_images[int(len(class_images) * (1 - perc)):]

            images += class_images

        if not out_of_distribution:
            self.original_indices = list(range(len(images)))

        else:
            self.original_indices = [-1] * len(images)

        if not test and not few_shot:
            random.shuffle(self.original_indices)
            images = [images[i] for i in self.original_indices]

        self.image_paths, self.labels = zip(*images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        original_index = self.original_indices[index]

        return image, label, original_index

    def sample(self):
        filtered = [
            (img, label, index) for img, label, index in zip(self.image_paths, self.labels, self.original_indices)
            if np.random.rand() < 0.5
        ]

        new_dataset = CustomDataset.__new__(CustomDataset)
        new_dataset.image_paths, new_dataset.labels, new_dataset.original_indices = zip(*filtered)
        new_dataset.transform = self.transform
        new_dataset.classes = self.classes
        new_dataset.class_labels = self.class_labels

        return new_dataset

    def join(self, second_dataset):
        new_dataset = CustomDataset.__new__(CustomDataset)
        new_dataset.image_paths = list(self.image_paths) + list(second_dataset.image_paths)
        new_dataset.labels = list(self.labels) + list(second_dataset.labels)
        new_dataset.original_indices = list(self.original_indices) + list(second_dataset.original_indices)
        new_dataset.transform = self.transform
        new_dataset.classes = self.classes
        new_dataset.class_labels = self.class_labels

        return new_dataset
