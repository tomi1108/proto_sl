import argparse
import os
import csv
from typing import Tuple, Dict, List

def create_directory(
    args: argparse.ArgumentParser,
    dir_path: str
) -> Tuple[str, str]:
    
    '''
    結果を記録するためのディレクトリを作成する関数
    既にディレクトリがあるときは何もしない
    '''
    
    result_path = args.results_path + dir_path
    accuracy_dir_path = result_path + 'accuracy/'
    distribution_dir_path = result_path + 'distribution/'
    loss_dir_path = result_path + 'loss/'

    if not os.path.exists(result_path):

        os.makedirs(result_path)
        print(f'Directory {result_path} created.')

    if not os.path.exists(accuracy_dir_path):

        os.makedirs(accuracy_dir_path)
        print(f'Directory {accuracy_dir_path} created.')

    if not os.path.exists(distribution_dir_path):

        os.makedirs(distribution_dir_path)
        print(f'Directory {distribution_dir_path} created.')

    if not os.path.exists(loss_dir_path):

        os.makedirs(loss_dir_path)
        print(f'Directory {loss_dir_path} created.')

    return accuracy_dir_path, distribution_dir_path, loss_dir_path
    

def set_output_file(
    args: argparse.ArgumentParser,
    acc_dir_path: str,
    loss_dir_path: str,
    file_name: str
) -> Tuple[str, str]:
    
    '''
    結果を記録するためのCSVファイルの準備
    ファイルへのパスを返す
    '''
    
    acc_file_path = acc_dir_path + file_name + '_acc.csv'
    loss_file_path = loss_dir_path + file_name + '_loss.csv'

    acc_header = ['Round', 'Accuracy']
    loss_header = ['Epoch', 'Loss']
    # headerが変わるアプローチが増えてきたら関数化する
    if args.proto_flag:
        loss_header += ['Proto Loss', 'Total Loss']
    
    with open(acc_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(acc_header)
    
    with open(loss_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loss_header)

    return acc_file_path, loss_file_path


def save_data_dist(
    num_classes: int,
    dist_dir_path: str,
    file_name: str,
    data_dist_dict: Dict[str, List[int]]
):
    
    dist_file_path = dist_dir_path + file_name + '_dist.csv'
    dist_header = ['Client'] + [ 'Class {}'.format(cls) for cls in range(num_classes) ]

    with open(dist_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(dist_header)

    for client_id, distribution in data_dist_dict.items():
        
        input_list = [client_id] + distribution

        with open(dist_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(input_list)