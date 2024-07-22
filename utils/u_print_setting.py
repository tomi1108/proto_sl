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
        train_mode = train_mode + '_SFL'
    if args.proto_flag == True:
        train_mode = train_mode + '_Proto'
    if args.self_kd_flag == True:
        train_mode = train_mode + '_Self-KD'
    
    print(" - train mode: ", train_mode)

    dict_name1 = train_mode
    dict_name2 = args.model_name + '_' + args.dataset_type
    file_name = str(datetime.today().date()) + '_B' + str(args.batch_size) + '_R' + str(args.num_rounds) + '_E' + str(args.num_epochs)

    print(" - total: {}\n".format(dict_name1+'_'+dict_name2+'_'+file_name))

    return dict_name1 + '/' + dict_name2, file_name + '_loss.csv', file_name + '_accuracy.csv'