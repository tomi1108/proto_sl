import argparse
import os
import csv

def print_setting(args: argparse.ArgumentParser):

    print("\n========== TRAIN SETTING ==========\n")

    # 使用するデータセットについて
    if args.dataset_type == 'cifar10':
        print(" - dataset: CIFAR10")

    # データ分布の設定方法について
    if args.data_partition == 0:
        print(" - data setting: IID")
        data_type = 'IID'
    elif args.data_partition == 1:
        print(" - data setting: Non-IID (class separation)")
        data_type = 'N-IID-ClassSep'
    elif args.data_partition == 2:
        print(" - data setting: Non-IID (Dirichlet beta=0.6)")
        data_type = 'N-IID-Diri-06'
    elif args.data_partition == 3:
        print(" - data setting: Non-IID (Dirichlet beta=0.3)")
        data_type = 'N-IID-Diri-03'
    elif args.data_partition == 4:
        print(" - data setting: Non-IID (Dirichlet beta=0.1)")
        data_type = 'N-IID-Diri-01'
    elif args.data_partition == 5:
        print(" - data setting: Non-IID (Dirichlet beta=0.05)")
        data_type = 'N-IID-Diri-005'

    # 学習方法について
    train_mode = 'SL'
    if args.fed_flag:
        train_mode = train_mode + '_SFL'
    if args.proto_flag:
        train_mode = train_mode + '_Proto'
    if args.con_flag:
        train_mode = train_mode + '_Con'
    if args.moco_flag:
        train_mode = train_mode + '_Moco'
    if args.self_kd_flag:
        train_mode = train_mode + '_Self-KD'
    if args.kd_flag:
        train_mode = train_mode + '_KD'
    if args.mkd_flag:
        train_mode = train_mode + '_MKD'
    if args.TiM_flag:
        train_mode = train_mode + '_TiM'
    if args.Mix_flag:
        train_mode = train_mode + '_Mix'
    if args.Mix_flag and args.Mix_s_flag:
        train_mode = 'SL_SFL_2Mix'
    
    print(" - train mode: ", train_mode)

    dict_name1 = train_mode
    dict_name2 = args.model_name + '_' + args.dataset_type
    dict_name3 = args.date

    file_name = data_type + '_B' + str(args.batch_size) + '_R' + str(args.num_rounds) + '_E' + str(args.num_epochs)

    print(" - total: {}\n".format(dict_name1+'_'+dict_name2+'_'+dict_name3+'_'+file_name))

    return dict_name1 + '/' + dict_name2 + '/' + dict_name3, file_name + '_loss.csv', file_name + '_accuracy.csv'

def fl_print_setting(args: argparse.ArgumentParser):

    print("\n========== TRAIN SETTING ==========\n")

    # 使用するデータセットについて
    if args.dataset_type == 'cifar10':
        print(" - dataset: CIFAR10")

    # データ分布の設定方法について
    if args.data_partition == 0:
        print(" - data setting: IID")
        data_type = 'IID'
    elif args.data_partition == 1:
        print(" - data setting: Non-IID (class separation)")
        data_type = 'N-IID-ClassSep'
    elif args.data_partition == 2:
        print(" - data setting: Non-IID (Dirichlet beta=0.6)")
        data_type = 'N-IID-Diri-06'
    elif args.data_partition == 3:
        print(" - data setting: Non-IID (Dirichlet beta=0.3)")
        data_type = 'N-IID-Diri-03'
    elif args.data_partition == 4:
        print(" - data setting: Non-IID (Dirichlet beta=0.1)")
        data_type = 'N-IID-Diri-01'
    elif args.data_partition == 5:
        print(" - data setting: Non-IID (Dirichlet beta=0.05)")
        data_type = 'N-IID-Diri-005'

    # 学習方法について
    train_mode = 'FL'
    
    print(" - train mode: ", train_mode)

    dict_name1 = train_mode
    dict_name2 = args.model_name + '_' + args.dataset_type
    dict_name3 = args.date

    file_name = data_type + '_B' + str(args.batch_size) + '_R' + str(args.num_rounds) + '_E' + str(args.num_epochs)

    print(" - total: {}\n".format(dict_name1+'_'+dict_name2+'_'+dict_name3+'_'+file_name))

    return dict_name1 + '/' + dict_name2 + '/' + dict_name3, file_name + '_loss.csv', file_name + '_accuracy.csv'

def create_directory(path: str):
    loss = '/loss/'
    accuracy = '/accuracy/'
    images = '/images/'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")
    if not os.path.exists(path+loss):
        os.makedirs(path+loss)
        print(f'Directory {path+loss} created')
    if not os.path.exists(path+accuracy):
        os.makedirs(path+accuracy)
        print(f'Directory {path+accuracy} created')
    if not os.path.exists(path+images):
        os.makedirs(path+images)
        print(f'Directory {path+images} created')

def fl_set_output_file(args: argparse.ArgumentParser, dict_path: str, loss_file: str, accuracy_file: str):

    path_to_loss_file = args.results_path + dict_path + '/loss/' + loss_file
    path_to_accuracy_file = args.results_path + dict_path + '/accuracy/' + accuracy_file

    acc_header = ['Round', 'Accuracy']
    loss_header = ['Epoch', 'Loss']

    with open(path_to_loss_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(loss_header)

    with open(path_to_accuracy_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(acc_header)

    return path_to_loss_file, path_to_accuracy_file