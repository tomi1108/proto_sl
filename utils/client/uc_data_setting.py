import argparse
import socket
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List

import utils.u_send_receive as u_sr
import utils.client.uc_send_receive as uc_sr

def data_setting(
    args: argparse.ArgumentParser,
    client_socket
):

    train_dataset, test_loader = create_dataset(args)
    num_classes, _ = get_train_data_info(args)
    
    index_to_index = u_sr.client(client_socket)

    indices_per_class = [ [] for _ in range(num_classes) ]
    for index, (_, label) in enumerate(train_dataset):
        indices_per_class[label].append(index)

    train_dataset_indices = []
    for cls in range(num_classes):

        idx_to_idx = index_to_index[cls]

        for iti in idx_to_idx:
            
            index = indices_per_class[cls][iti]
            train_dataset_indices.append(index)
    
    train_dataset = Subset(train_dataset, train_dataset_indices)

    class_counts = [ 0 for _ in range(num_classes) ]
    for idx in train_dataset.indices:
        target = train_dataset.dataset.targets[idx]
        class_counts[target] += 1
    print("data size for each class")
    print(class_counts)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.fed_flag:
        iteration_info = {'num_train_iterations': len(train_loader)}
    else:
        iteration_info = {'num_train_iterations': len(train_loader), 'num_test_iterations': len(test_loader)}

    u_sr.client(client_socket, iteration_info)

    return train_loader, test_loader


def create_dataset(
    args: argparse.ArgumentParser
):

    '''
    指定したデータセットを取得して返すための関数
    モデル集約をしないとき (fed_flag==False) は、テストローダも作成して返す
    '''

    test_loader = None # args.fed_flagがTrueならNoneのまま返す

    if args.dataset_type == 'cifar10':
        
        train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        if args.fed_flag == False:
            test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    
    elif args.dataset_type == 'mnist':

        train_dataset = dsets.MNIST(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        if args.fed_flag == False:
            test_dataset = dsets.MNIST(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    
    if args.fed_flag == False:
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False) 
    
    return train_dataset, test_loader


def get_train_data_info(
    args: argparse.ArgumentParser
) -> Tuple[int, List[int]]:

    '''
    学習データセットのクラス数と、クラスごとのサンプル数を取得する関数
    現状の実装では、クラスごとにサンプル数が異ならないことを前提としている
    '''

    if args.dataset_type == 'cifar10':

        num_classes = 10
        data_size_per_class = [ 5000 for _ in range(num_classes) ]

    elif args.dataset_type == 'cifar100':

        num_classes = 100
        data_size_per_class = [ 500 for _ in range(num_classes) ]
    
    return num_classes, data_size_per_class


######################################################################################################
# クライアントごとにデータ数が異なるセッティング

def set_data_dist(
    args: argparse.ArgumentParser,
    client_socket: socket.socket
):
    
    train_dataset, test_loader = create_dataset(args)
    num_classes, _ = get_train_data_info(args)
    idx_to_idx_per_class = uc_sr.receive(client_socket)
    fed_flag = args.fed_flag

    # クラスごとにデータのインデックスをまとめている
    indices_per_class = [ [] for _ in range(num_classes) ]
    for index, (_, label) in enumerate(train_dataset):
        indices_per_class[label].append(index)
    
    # 使用するデータのインデックスをまとめたリスト
    train_dataset_indices = []
    for cls in range(num_classes):

        idx_to_idx = idx_to_idx_per_class[cls]

        for iti in idx_to_idx:

            idx = indices_per_class[cls][iti]
            train_dataset_indices.append(idx)
    
    train_dataset = Subset(train_dataset, train_dataset_indices)

    # クラスごとのデータ数を表示
    class_counts = [ 0 for _ in range(num_classes) ]
    for idx in train_dataset.indices:
        target = train_dataset.dataset.targets[idx]
        class_counts[target] += 1
    print("data size for each class")
    print(class_counts)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if not fed_flag:
        # テストイテレーションをサーバに送信するコードを実装
        test_loader = None

    return train_loader, test_loader    