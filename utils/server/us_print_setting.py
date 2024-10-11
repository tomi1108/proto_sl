import argparse
from typing import Tuple

def print_setting(
    args: argparse.ArgumentParser
) -> Tuple[str, str]:
    
    '''
    学習の設定を表示するための関数
    学習結果を記録するディレクトリ名とファイル名を返す
    '''

    model_name = args.model_name
    dataset = args.dataset_type
    dist_type = 'alpha' + str(args.alpha)
    train_mode = named_result_dir(args)
    date = args.date
    round = 'R' + str(args.num_rounds)
    epoch = 'E' + str(args.num_epochs) 
    batch = 'B' + str(args.batch_size)
    
    print("\n========== TRAIN SETTING ==========\n")

    print(f' - model: {model_name}')
    print(f' - dataset: {dataset}')
    print(f' - data distribution: Dirichlet {dist_type}')
    print(f' - train mode: {train_mode}')

    dir_name1 = train_mode
    dir_name2 = model_name + '_' + dataset
    dir_name3 = date
    
    result_directory = dir_name1 + '/' + dir_name2 + '/' + dir_name3 + '/'
    file_name = round + '_' + epoch + '_'  + batch + '_'  + dist_type

    print(f' - total: {result_directory + file_name} ')

    return result_directory, file_name


def named_result_dir(
    args: argparse.ArgumentParser
) -> str:
    
    train_mode = 'SFL'

    if args.proto_flag:
        train_mode = 'Proto_' + train_mode
    
    if args.con_flag:
        train_mode = 'MOON_' + train_mode

    if args.mkd_flag:
        train_mode = 'MKD_' + train_mode
    
    return train_mode