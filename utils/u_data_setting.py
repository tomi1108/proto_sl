import argparse
import random
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.u_send_receive as u_sr
from torch.utils.data import DataLoader, Subset

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


def client_data_setting(
    args: argparse.ArgumentParser,
    client_socket
):

    train_dataset, test_loader = create_dataset(args)
    num_classes, data_size_per_class = get_train_data_info(args)
    
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


def get_train_data_info(
    args: argparse.ArgumentParser
):

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


def create_noniid(
    num_clients: int,
    num_classes: int,
    max_data_size_per_client: int,
    data_size_per_class: list,
    dirichlet_dist: list
):

    '''
    Dirichlet分布からnon-IID設定を作成する関数
    各クライアントが持つデータ数の合計は、元のデータセットの1クラス当たりのサンプル数と同じ
    reminder_sizeのおかげで、合計のデータ数がクライアントごとに変わることはない
    '''
    
    data_size_list = []

    for clt in range(num_clients):

        reminder_size = max_data_size_per_client
        counts_per_class = []

        for cls in range(num_classes):

            if cls < num_classes - 1:
                
                counts = int( data_size_per_class[cls] * dirichlet_dist[clt][cls] )
                counts_per_class.append(counts)
            
            elif cls == num_classes - 1:

                counts = reminder_size
                counts_per_class.append(counts)
            
            reminder_size = reminder_size - counts
        
        if reminder_size == 0:

            data_size_list.append(counts_per_class)

        else:

            raise ValueError("There is a problem with the number of data for each client.")

    return data_size_list   


def create_indices(
    num_clients: int,
    num_classes: int,
    sampling_counts: list,
    max_size: int
):

    '''
    クライアントごとの、使用する各クラスのデータのインデックスを取得
     ⇒ CIFAR10なら 0 ~ 4999 のインデックス
    クライアント側は、クラスごとにデータのインデックスをリスト化する
    そして、この関数によって作成されるリストが示すインデックスを参照し、そこで指定されているインデックスのデータを使用する
    '''

    sampled_indices_per_client = []

    for clt in range(num_clients):

        sampled_indices = []

        for cls in range(num_classes):

            num_samples = sampling_counts[clt][cls]
            
            if num_samples > 0:
                
                sampled_indices.append(np.random.choice(max_size, num_samples, replace=False))

            else:

                sampled_indices.append([])
            
        sampled_indices_per_client.append(sampled_indices)
    
    return sampled_indices_per_client
    

def check_iterations(
    iteration_info_dict: dict,
    key: str
):
    
    num_iterations = iteration_info_dict['Client 1'][key]

    for iteration_info in iteration_info_dict.values():

        if iteration_info[key] != num_iterations:
            raise ValueError(f'The number of iterations is different. (key: {key})')
    
    return num_iterations


def create_test_loader_in_server(
    args: argparse.ArgumentParser
):
    
    '''
    モデル集約をする場合にサーバでテストローダを作成するための関数
    '''
    
    dataset_type = args.dataset_type
    root = args.dataset_path
    batch_size = args.batch_size

    if dataset_type == 'cifar10':

        test_dataset = dsets.CIFAR10(root=root, train=False, transform=transforms.ToTensor(), download=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return test_loader


def server_data_setting(
    args: argparse.ArgumentParser,
    connections: dict
):
    
    num_classes, data_size_per_class = get_train_data_info(args)
    num_clients = args.num_clients
    alpha = args.alpha
    seed = args.seed
    fed_flag = args.fed_flag

    alpha = np.asarray(alpha)
    alpha = np.repeat(alpha, num_classes)

    dirichlet_dist = np.random.default_rng(seed).dirichlet(
        alpha=alpha,
        size=num_clients
    )

    max_data_size_per_client = int(data_size_per_class[0]) # クラスごとにデータ数が違うときは最小値でいいかも

    data_size_list = create_noniid(
                        num_clients,
                        num_classes,
                        max_data_size_per_client,
                        data_size_per_class,
                        dirichlet_dist
                        )

    indices_per_class_per_client = create_indices(
                                        num_clients,
                                        num_classes,
                                        data_size_list,
                                        max_data_size_per_client
                                        )
    
    for indices, connection in zip(indices_per_class_per_client, connections.values()):
        u_sr.server(connection, b"SEND", indices)
    

    iteration_info_dict = {}

    for client_id, connection in connections.items():
        iteration_info_dict[client_id] = u_sr.server(connection, b"REQUEST")

    num_train_iterations = check_iterations(iteration_info_dict, key='num_train_iterations')

    if fed_flag:

        test_loader = create_test_loader_in_server(args)

        return test_loader, num_classes, num_train_iterations, None   
        
    else:

        num_test_iterations = check_iterations(iteration_info_dict, key='num_test_iterations')

        return None, num_classes, num_train_iterations, num_test_iterations


# def server_data_setting(args: argparse.ArgumentParser, connections: dict):

#     if args.dataset_type == 'cifar10':
#         train_dataset = dsets.CIFAR10(root=args.dataset_path, train=True, transform=transforms.ToTensor(), download=True)
#     num_class = len(train_dataset.classes)

#     if args.data_partition == 0: # IIDを指定
#         client1_info = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
#         client2_info = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
#     elif args.data_partition == 1: # Non-IID（クラス分割）を指定
#         # split_list = split_class(args, num_class)
#         # client1_info = split_list[0]
#         # client2_info = split_list[1]
#         client1_info = [5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]
#     elif args.data_partition == 2: # Non-IID（Dirichlet beta=0.6）を指定
#         client1_info = [1356, 3488, 3932, 4581, 2311, 2772, 1921, 13, 4597, 29]
#         client2_info = [3644, 1512, 1068, 419, 2689, 2228, 3079, 4987, 403, 4971]
#     elif args.data_partition == 3: # Non-IID（Dirichlet beta=0.3）を指定
#         client1_info = [2755, 3706, 4298, 3029, 1, 1918, 982, 1159, 3958, 3195]
#         client2_info = [2245, 1294, 702, 1971, 4999, 3082, 4018, 3841, 1042, 1805]
#     elif args.data_partition == 4: # Non-IID (Dirichlet beta=0.1) を指定
#         client1_info = [4665, 570, 4406, 0, 3604, 2971, 583, 3633, 4, 4564]
#         client2_info = [335, 4430, 594, 5000, 1396, 2029, 4417, 1367, 4996, 436]
#     elif args.data_partition == 5: # Non-IID (Dirichlet beta=0.05) を指定
#         client1_info = [0, 2757, 4432, 4566, 0, 4444, 4394, 2, 0, 4405]
#         client2_info = [5000, 2243, 568, 434, 5000, 556, 606, 4998, 5000, 595]   
    
#     class_indices = [[] for _ in range(num_class)]
#     for idx, (_, label) in enumerate(train_dataset):
#         class_indices[label].append(idx)
    
#     # クライアント数が増えてもここを変えれば大丈夫    
#     client1_list = []
#     client2_list = []
#     for class_id, info in enumerate(client1_info):
#         client1_list.append(class_indices[class_id][:info])
#         client2_list.append(class_indices[class_id][info:5000])
    
#     client1_idx = [idx for sublist in client1_list for idx in sublist]
#     client2_idx = [idx for sublist in client2_list for idx in sublist]

#     u_sr.server(connections['Client 1'], b"SEND", client1_idx)
#     u_sr.server(connections['Client 2'], b"SEND", client2_idx)

#     iteration_info_list = {}
#     if args.fed_flag == True:
#         for client_id, connection in connections.items():
#             iteration_info_list[client_id] = u_sr.server(connection, b"REQUEST")
#         for iteration_info in iteration_info_list.values():
#             if iteration_info['num_iterations'] != iteration_info_list['Client 1']['num_iterations']:
#                 print("The number of iterations is different")
#                 raise ValueError
        
#         num_iterations = iteration_info_list['Client 1']['num_iterations']

#         if args.dataset_type == 'cifar10':
#             test_dataset = dsets.CIFAR10(root=args.dataset_path, train=False, transform=transforms.ToTensor(), download=True)
#         test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False)
        
#         return test_loader, num_class, num_iterations, None
#     else:
#         for client_id, connection in connections.items():
#             iteration_info_list[client_id] = u_sr.server(connection, b"REQUEST")
#         for iteration_info in iteration_info_list.values():
#             if iteration_info['num_iterations'] != iteration_info_list['Client 1']['num_iterations']:
#                 print("The number of iterations is different")
#                 raise ValueError
#             if iteration_info['num_test_iterations'] != iteration_info_list['Client 1']['num_test_iterations']:
#                 print("The number of test iterations is different.")
#                 raise ValueError
        
#         num_iterations = iteration_info_list['Client 1']['num_iterations']
#         num_test_iterations = iteration_info_list['Client 1']['num_test_iterations']

#         return None, num_class, num_iterations, num_test_iterations


    
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