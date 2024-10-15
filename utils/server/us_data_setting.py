import argparse
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from typing import Tuple, List
from torch.utils.data import DataLoader

import utils.u_send_receive as u_sr
import utils.server.us_send_receive as us_sr



def data_setting(
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

    # max_data_size_per_client = int(data_size_per_class[0]) # クラスごとにデータ数が違うときは最小値でいいかも
    max_data_size_per_client = int(sum(data_size_per_class) / num_clients)

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
                                        max_data_size_per_client,
                                        data_size_per_class
                                        )
    
    for indices, connection in zip(indices_per_class_per_client, connections.values()):
        u_sr.server(connection, b"SEND", indices)
    

    iteration_info_dict = {}

    for client_id, connection in connections.items():
        iteration_info_dict[client_id] = u_sr.server(connection, b"REQUEST")

    num_train_iterations = check_iterations(iteration_info_dict, key='num_train_iterations')

    if fed_flag:

        test_loader = create_test_loader(args)

        return test_loader, num_classes, num_train_iterations, None   
        
    else:

        num_test_iterations = check_iterations(iteration_info_dict, key='num_test_iterations')

        return None, num_classes, num_train_iterations, num_test_iterations


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


def create_noniid(
    num_clients: int,
    num_classes: int,
    max_data_size_per_client: int,
    data_size_per_class: list,
    dirichlet_dist: list
) -> List[List[int]]:

    '''
    Dirichlet分布からnon-IID設定を作成する関数
    各クライアントが持つデータ数の合計は、元のデータセットの1クラス当たりのサンプル数と同じ
    reminder_sizeのおかげで、合計のデータ数がクライアントごとに変わることはない

    num_clients: クライアント数
    num_classes: クラス数
    max_data_size_per_client: クライアントが持てる最大データ数 (最もデータ数が少ないクラスのデータ数)
    data_size_per_class: クラスごとのデータサイズ
    dirichlet_dist: Dirichlet分布で生成したデータ分布 (クライアント数 X クラス数)
    '''
    
    data_size_list = []

    for clt in range(num_clients):

        reminder_size = max_data_size_per_client
        counts_per_class = []

        for cls in range(num_classes):

            if cls < num_classes - 1:
                
                # counts = int( data_size_per_class[cls] * dirichlet_dist[clt][cls] )
                counts = int( max_data_size_per_client * dirichlet_dist[clt][cls])
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
    max_size: int,
    data_size_per_class: List[int]
) -> List[List[List[int]]]:

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
                
                index_list = []
                data_size = data_size_per_class[cls]
                
                if num_samples > data_size:
                    
                    times, amari = divmod(num_samples, data_size)
                    for _ in range(times):
                        index_list.extend(range(data_size))
                    num_samples = amari

                index_list.extend(np.random.choice(data_size, num_samples, replace=False))
                sampled_indices.append(index_list)

            else:

                sampled_indices.append([])
            
        sampled_indices_per_client.append(sampled_indices)
    
    return sampled_indices_per_client


def check_iterations(
    iteration_info_dict: dict,
    key: str
) -> int:
    
    num_iterations = iteration_info_dict['Client 1'][key]

    for iteration_info in iteration_info_dict.values():

        if iteration_info[key] != num_iterations:
            raise ValueError(f'The number of iterations is different. (key: {iteration_info[key]})')
    
    return num_iterations


def create_test_loader(
    args: argparse.ArgumentParser
) -> DataLoader:
    
    '''
    モデル集約をする場合にサーバでテストローダを作成するための関数
    '''
    
    dataset_type = args.dataset_type
    dataset_path = args.dataset_path
    batch_size = args.batch_size

    if dataset_type == 'cifar10':

        test_dataset = dsets.CIFAR10(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    
    elif dataset_type == 'cifar100':

        test_dataset = dsets.CIFAR100(root=dataset_path, train=False, transform=transforms.ToTensor(), download=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return test_loader


######################################################################################################
# クライアントごとにデータ数が異なるセッティング

def set_client_data_dist(
    args: argparse.ArgumentParser,
    connections: dict
):
    
    num_classes, data_size_per_class = get_train_data_info(args)
    num_clients = args.num_clients
    batch_size = args.batch_size
    alpha = args.alpha
    seed = args.seed
    fed_flag = args.fed_flag

    alpha = np.asarray(alpha)
    alpha = np.repeat(alpha, num_clients)

    # クラス数 × クライアント数 -> クラスごとに、どのようにデータを割り振るかを示したリスト
    dirichlet_dist = np.random.default_rng(seed).dirichlet(
        alpha=alpha,
        size=num_classes
    )

    # ここで、run_sl.shで設定した値に従って、実行する関数を変更するようにする -> iid なども簡単に設定可能になる
    indices_of_clients, data_size_per_client = create_non_IID(
        num_clients,
        num_classes,
        data_size_per_class,
        dirichlet_dist
    )

    total_train_data_size = sum(data_size_per_class)
    num_train_iterations = total_train_data_size // batch_size
    weighs_in_model_aggregation = [ x / total_train_data_size for x in data_size_per_client]

    for indices, connection in zip(indices_of_clients, connections.values()):
        us_sr.send(connection, indices)
    
    if fed_flag:

        test_loader = create_test_loader(args)
        return test_loader, num_classes, num_train_iterations, None, weighs_in_model_aggregation

    else:

        # テストイテレーションを取得するコードを実装
        num_test_iterations = 0
        return None, num_classes, num_test_iterations, num_test_iterations, weighs_in_model_aggregation


def create_non_IID(
    num_clients: int,
    num_classes: int,
    data_size_per_class: List[int],
    dirichlet_dist: List[List[float]]
) -> List[List[int]]:
    
    '''
    各クライアントが、各クラスのデータをどのように持つかを決める関数
    クライアントごとに、保持するデータ数は異なるように設定
    クライアント数 x クラス数 となるリストを返す
    '''

    data_size_per_client = [ 0 for _ in range(num_clients) ]
    total_indices_list = [ [] for _ in range(num_clients)]

    for cls in range(num_classes):

        class_data_size = data_size_per_class[cls] # クラスclsのデータサイズ
        index_pointer = 0

        for clt in range(num_clients):

            # class_data_size_of_clt == 0 の場合、からのリスト [] が格納される、はず
            if clt < num_clients-1:

                class_data_size_of_clt = int( class_data_size * dirichlet_dist[cls][clt] )
                total_indices_list[clt].append(list(range(index_pointer, index_pointer + class_data_size_of_clt)))

                index_pointer += class_data_size_of_clt

            elif clt == num_clients-1:

                class_data_size_of_clt = int( class_data_size - index_pointer )
                total_indices_list[clt].append(list(range(index_pointer, index_pointer + class_data_size_of_clt)))
            
            data_size_per_client[clt] += class_data_size_of_clt
    
    return total_indices_list, data_size_per_client