import argparse
import random
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.u_send_receive as u_sr

from torch.utils.data import DataLoader, Subset

def client_data_setting(args: argparse.ArgumentParser, client_socket):
    
    # CIFAR10で学習
    if args.dataset_type == 'cifar10':
        train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
        if args.fed_flag == False:
            test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)

    # クラス数をサーバに送信
    num_classes = len(train_dataset.classes)
    u_sr.client(client_socket, num_classes)

    # クラス分布情報をサーバから受信
    class_distribution = u_sr.client(client_socket)
    if class_distribution == None:
        print("IID Setting!")
    else:
        print("Non-IID Setting!")

    # クラス分布情報に従ってデータセットを準備
    # IID設定の場合のデータ設定
    if args.data_partition == 0:
        
        targets = np.array(train_dataset.targets)
        indices_per_class = {i: np.where(targets == i)[0] for i in range(num_classes)}

        reduced_indices = []
        for indices in indices_per_class.values():
            np.random.shuffle(indices)
            reduced_indices.extend(indices[:int(len(indices) * (1 / args.num_clients))])

        train_dataset = Subset(train_dataset, reduced_indices)

    # Non-IID設定（クラス分割）
    elif args.data_partition == 1: 

        train_dataset = [data for data in train_dataset if data[1] in class_distribution]
    
    # Non-IID（Dirichlet分布）
    elif args.data_partition in [2, 3, 4, 5]: 
        
        targets = np.array(train_dataset.targets)
        indices_per_class = {i: np.where(targets == i)[0] for i in range(num_classes)}
        
        selected_data_index = []
        for class_idx, data_size in enumerate(class_distribution):
            selected_index = random.sample(list(indices_per_class[class_idx]), data_size)
            selected_data_index.extend(selected_index)
        
        train_dataset = Subset(train_dataset, selected_data_index)

    # 各クラスのデータ数を出力
    class_counts = [0 for _ in range(num_classes)]
    for _, target in train_dataset:
        class_counts[target] += 1
    print("data size for each class")
    print(class_counts)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.fed_flag == True:
        iteration_info = {'num_iterations': len(train_loader)}
        u_sr.client(client_socket, iteration_info)
        return train_loader, None
    else:
        iteration_info = {'num_iterations': len(train_loader), 'num_test_iteration': len(test_loader)}
        u_sr.client(client_socket, iteration_info)
        return train_loader, test_loader


def server_data_setting(args: argparse.ArgumentParser, connections: dict):

    class_list = {}
    for client_id, connection in connections.items():
        class_list[client_id] = u_sr.server(connection, b"REQUEST")
    for client_id, num_class in class_list.items():
        if num_class != class_list['Client 1']:
            print("Number of class is different ({}).".format(num_class))
            raise ValueError

    num_class = class_list['Client 1']
        
    # 現状はargs.num_clients==2であることが前提の実装
    if args.data_partition == 0: # IIDを指定
        client1_info = None
        client2_info = None
    elif args.data_partition == 1: # Non-IID（クラス分割）を指定
        split_list = split_class(args, num_class)
        client1_info = split_list[0]
        client2_info = split_list[1]
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

    u_sr.server(connections['Client 1'], b"SEND", client1_info)
    u_sr.server(connections['Client 2'], b"SEND", client2_info)
    
    iteration_info_list = {}
    if args.fed_flag == True:
        for client_id, connection in connections.items():
            iteration_info_list[client_id] = u_sr.server(connection, b"REQUEST")
        for iteration_info in iteration_info_list.values():
            if iteration_info['num_iterations'] != iteration_info_list['Client 1']['num_iterations']:
                print("The number of iterations is different")
                raise ValueError
        
        num_iterations = iteration_info_list['Client 1']['num_iterations']

        if args.dataset_type == 'cifar10':
            test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
        
        return test_loader, num_class, num_iterations, None
    else:
        for client_id, connection in connections.items():
            iteration_info_list[client_id] = u_sr.server(connection, b"REQUEST")
        for iteration_info in iteration_info_list.values():
            if iteration_info['num_iterations'] != iteration_info_list['Client 1']['num_iterations']:
                print("The number of iterations is different")
                raise ValueError
            if iteration_info['num_test_iterations'] != iteration_info_list['Client 1']['num_test_iterations']:
                print("The number of test iterations is different.")
                raise ValueError
        
        num_iterations = iteration_info_list['Client 1']['num_iterations']
        num_test_iterations = iteration_info_list['Client 1']['num_test_iterations']

        return None, num_class, num_iterations, num_test_iterations
    
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

