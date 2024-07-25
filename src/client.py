import sys
sys.path.append('./../')

import random
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import socket
import numpy as np
import argparse
from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_send_receive as u_sr
import utils.u_data_setting as u_dset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_connection(args: argparse.ArgumentParser):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', args.port_number)
    client_socket.connect(server_address)
    client_id = u_sr.client(client_socket)

    return client_socket, client_id

def set_model(args: argparse.ArgumentParser, client_socket: socket.socket):
    client_model = u_sr.client(client_socket).to(device)
    optimizer = torch.optim.SGD(params=client_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
    )
    return client_model, optimizer

def main(args: argparse.ArgumentParser):

    # シードを設定し再現性を持たせる
    set_seed(args.seed)
    
    # サーバと通信開始
    client_socket, client_id = set_connection(args)
    print("\n========== Client {} ==========\n".format(client_id))

    # データローダの作成
    train_loader, test_loader = u_dset.client_data_setting(args, client_socket)
    
    # クライアントモデルを受信
    client_model, optimizer = set_model(args, client_socket)
    client_model = client_model.to(device)


    # 学習開始
    for round in range(args.num_rounds):

        client_model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):

            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for i, (images, labels) in enumerate(tqdm(train_loader)):

                send_data_dict = {'smashed_data': None, 'labels': None}
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                smashed_data = client_model(images)
                send_data_dict['smashed_data'] = smashed_data.to('cpu')
                send_data_dict['labels'] = labels.to('cpu')
                u_sr.client(client_socket, send_data_dict)

                gradients = u_sr.client(client_socket).to(device)
                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(gradient=smashed_data.grad)
                optimizer.step()
        
        if args.fed_flag == True:
            # 平均化
            client_model = client_model.to('cpu')
            u_sr.client(client_socket, client_model)
            client_model.load_state_dict(u_sr.client(client_socket).state_dict())
            client_model = client_model.to(device)
        else:
            # 各クライアントでテストを行う
            pass


    # 通信終了
    client_socket.close()

if __name__ == '__main__':

    args = u_parser.arg_parser()
    print(args)
    main(args)