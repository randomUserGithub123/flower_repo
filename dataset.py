import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader, Subset

import numpy as np

def get_mnist(data_path: str = './data'):
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2024))

    trainloaders = []
    valloaders = []
    for tr_set in trainsets:
        num_total = len(tr_set)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(tr_set, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=120)

    return trainloaders, valloaders, testloader

def prepare_dataset_nonIID(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset = get_mnist()

    targets = np.array(trainset.targets)
    sorted_indices = np.argsort(targets)
    sorted_targets = targets[sorted_indices]

    class_indices = []
    for class_label in range(10):
        class_indices.append(sorted_indices[sorted_targets == class_label])

    shards = []
    num_shards_per_class = 2
    for indices in class_indices:
        shard_size = len(indices) // num_shards_per_class
        for i in range(num_shards_per_class):
            shards.append(indices[i * shard_size: (i + 1) * shard_size])
    
    np.random.seed(2024)
    np.random.shuffle(shards)
    
    client_indices = []
    used_labels = set()
    shard_label_map = {i: set(targets[shard]) for i, shard in enumerate(shards)}

    for i in range(0, num_partitions, 2):
        shard_pair = []
        for shard_idx, labels in shard_label_map.items():
            if not labels & used_labels:
                shard_pair.append(shard_idx)
                used_labels.update(labels)
            if len(shard_pair) == 2:
                break
        
        if len(shard_pair) != 2:
            raise ValueError("Not enough unique shards to assign to each client pair.")

        client_shards = [shards[shard_pair[0]], shards[shard_pair[1]]]
        client_data = np.concatenate(client_shards)
        client_indices.append(client_data)
        client_indices.append(client_data)

    trainloaders = []
    valloaders = []
    for indices in client_indices:
        client_subset = Subset(trainset, indices)
        
        num_total = len(client_subset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(client_subset, [num_train, num_val], torch.Generator().manual_seed(2024))
        train_labels = [label for _, label in for_train]
        print(set(train_labels))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=True, num_workers=2))

    testloader = DataLoader(testset, batch_size=120)

    return trainloaders, valloaders, testloader
