import argparse
import os
import csv
from typing import Tuple

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
    loss_dir_path = result_path + 'loss/'

    if not os.path.exists(result_path):

        os.makedirs(result_path)
        print(f'Directory {result_path} created.')

    if not os.path.exists(accuracy_dir_path):

        os.makedirs(accuracy_dir_path)
        print(f'Directory {accuracy_dir_path} created.')

    if not os.path.exists(loss_dir_path):

        os.makedirs(loss_dir_path)
        print(f'Directory {loss_dir_path} created.')

    return accuracy_dir_path, loss_dir_path
    

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
