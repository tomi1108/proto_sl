import argparse
import random
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.u_send_receive as u_sr
from torch.utils.data import DataLoader, Subset


    
def split_class(args: argparse.ArgumentParser, num_class: int):
    class_list = list(range(num_class))
    chunk_size = num_class // args.num_clients
    remainder = num_class % args.num_clients

    split_list = []
    start = 0
    for i in range(args.num_clients):
        end = start + chunk_size + (1 if i < remainder else 0)
        split_list.append(class_list[start:end])
        start = end
    
    return split_list

def fl_client_setting(args: argparse.ArgumentParser, client_socket):

    if args.dataset_type == 'cifar10':
        train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    num_class = len(train_dataset.classes)
    train_dataset_idx = u_sr.client(client_socket)
    train_dataset = Subset(train_dataset, train_dataset_idx)
    
    class_counts = [0 for _ in range(num_class)]
    for _, target in train_dataset:
        class_counts[target] += 1
    print("data size for each class")
    print(class_counts)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader


def fl_server_setting(args: argparse.ArgumentParser, connections: dict):

    if args.dataset_type == 'cifar10':
        train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    num_class = len(train_dataset.classes)

    if args.data_partition == 0: # IIDを指定
        client1_info = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
        client2_info = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
    elif args.data_partition == 1: # Non-IID（クラス分割）を指定
        # split_list = split_class(args, num_class)
        # client1_info = split_list[0]
        # client2_info = split_list[1]
        client1_info = [5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]
    elif args.data_partition == 2: # Non-IID（Dirichlet beta=0.6）を指定
        client1_info = [1356, 3488, 3932, 4581, 2311, 2772, 1921, 13, 4597, 29]
        client2_info = [3644, 1512, 1068, 419, 2689, 2228, 3079, 4987, 403, 4971]
    elif args.data_partition == 3: # Non-IID（Dirichlet beta=0.3）を指定
        client1_info = [2755, 3706, 4298, 3029, 1, 1918, 982, 1159, 3958, 3195]
        client2_info = [2245, 1294, 702, 1971, 4999, 3082, 4018, 3841, 1042, 1805]
    elif args.data_partition == 4: # Non-IID (Dirichlet beta=0.1) を指定
        client1_info = [4665, 570, 4406, 0, 3604, 2971, 583, 3633, 4, 4564]
        client2_info = [335, 4430, 594, 5000, 1396, 2029, 4417, 1367, 4996, 436]
    elif args.data_partition == 5: # Non-IID (Dirichlet beta=0.05) を指定
        client1_info = [0, 2757, 4432, 4566, 0, 4444, 4394, 2, 0, 4405]
        client2_info = [5000, 2243, 568, 434, 5000, 556, 606, 4998, 5000, 595]   
    
    class_indices = [[] for _ in range(num_class)]
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    # クライアント数が増えてもここを変えれば大丈夫    
    client1_list = []
    client2_list = []
    for class_id, info in enumerate(client1_info):
        client1_list.append(class_indices[class_id][:info])
        client2_list.append(class_indices[class_id][info:5000])
    
    client1_idx = [idx for sublist in client1_list for idx in sublist]
    client2_idx = [idx for sublist in client2_list for idx in sublist]

    u_sr.server(connections['Client 1'], b"SEND", client1_idx)
    u_sr.server(connections['Client 2'], b"SEND", client2_idx)

    if args.dataset_type == 'cifar10':
        test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
    
    return test_loader, num_class