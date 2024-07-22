import argparse
from datetime import datetime

def print_setting(args: argparse.ArgumentParser):

    print("\n========== TRAIN SETTING ==========\n")

    # 使用するデータセットについて
    if args.dataset_type == 'cifar10':
        print(" - dataset: CIFAR10")

    # データ分布の設定方法について
    if args.data_partition == 0:
        print(" - data setting: IID")
    elif args.data_partition == 1:
        print(" - data setting: Non-IID (class separation)")
    elif args.data_partition == 2:
        print(" - data setting: Non-IID (Dirichlet beta=0.6)")
    elif args.data_partition == 3:
        print(" - data setting: Non-IID (Dirichlet beta=0.3)")

    # 学習方法について
    train_mode = 'SL'
    if args.fed_flag == True:
        train_mode = train_mode + '_splitfed'
    if args.proto_flag == True:
        train_mode = train_mode + '_proto'
    if args.self_kd_flag == True:
        train_mode = train_mode + '_self-kd'
    
    print(" - train mode: ", train_mode)

    total_infomation = str(datetime.today().date()) + '_' + train_mode + '_' + args.dataset_type
    print(" - total: {}\n".format(total_infomation))

    return total_infomation