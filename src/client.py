import sys
sys.path.append('./../')

import random
import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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
import utils.u_momentum_contrast as u_moco

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
pid = os.getpid()
print(f'The process ID (PID) is: {pid}')
time.sleep(5)

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

    # MOON用のモデルを定義
    if args.con_flag == True:
        previous_client_model = None
        global_client_model = None
        projection_head = u_sr.client(client_socket).to(device)
        criterion = nn.CrossEntropyLoss()
        cos = nn.CosineSimilarity(dim=-1)

    # 双方向知識蒸留用モデルを定義
    if args.mkd_flag:
        global_client_model = copy.deepcopy(client_model)
        global_optimizer = torch.optim.SGD(params=global_client_model.parameters(),
                                           lr=args.lr,
                                           momentum=args.momentum,
                                           weight_decay=args.weight_decay
                                           )
        projection_head = u_sr.client(client_socket).to(device)
        mkd_criterion = nn.KLDivLoss(reduction='batchmean')
    
    if args.moco_flag:
        global_client_model = copy.deepcopy(client_model)
        for param in global_client_model.parameters():
            param.requires_grad = False
        projection_head = u_sr.client(client_socket).to(device)
        Moco = u_moco.Moco(args)
        criterion = nn.CrossEntropyLoss()

    # 学習開始
    for round in range(args.num_rounds):

        client_model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):

            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for i, (images, labels) in enumerate(tqdm(train_loader)):

                send_data_dict = {'smashed_data': None, 'labels': None}

                if args.moco_flag:
                    images_q = images[0].to(device)
                    images_k = images[1].to(device)
                    labels = labels.to(device)

                    smashed_data = client_model(images_q)
                else:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    smashed_data = client_model(images)
                
                send_data_dict['smashed_data'] = smashed_data.to('cpu')
                send_data_dict['labels'] = labels.to('cpu')
                u_sr.client(client_socket, send_data_dict)

                if args.moco_flag:
                    optimizer.zero_grad()

                    moco_outs, moco_labels = Moco.compute_moco(images_q, images_k, client_model, global_client_model, projection_head)
                    moco_loss = criterion(moco_outs, moco_labels)
                    moco_loss.backward(retain_graph=True)
                    grads1 = [param.grad.clone() for param in client_model.parameters()]


                if args.con_flag == True and round > 0:

                    optimizer.zero_grad()
                    
                    smashed_output = projection_head(smashed_data.to(device))
                    previous_output = projection_head(previous_client_model(images))
                    global_output = projection_head(global_client_model(images))

                    pos = cos(smashed_output, global_output)
                    logits = pos.reshape(-1, 1)
                    neg = cos(smashed_output, previous_output)
                    logits = torch.cat((logits, neg.reshape(-1, 1)), dim=1)

                    logits /= 0.5
                    logits_labels = torch.zeros(images.size(0)).to(device).long()

                    con_loss = criterion(logits, logits_labels)
                    con_loss.backward(retain_graph=True)
                    grads1 = [param.grad.clone() for param in client_model.parameters()]

                    if i % 100 == 0:
                        print('con_loss: ', con_loss.item())
                
                if args.mkd_flag == True and round > 0:

                    optimizer.zero_grad()
                    global_optimizer.zero_grad()

                    smashed_output = projection_head(smashed_data.to(device))
                    global_output = projection_head(global_client_model(images))

                    client_loss = mkd_criterion(F.log_softmax(smashed_output / 2.0, dim=1), 
                                                F.softmax(global_output.detach() / 2.0, dim=1)) * (2.0 * 2.0)
                    global_loss = mkd_criterion(F.log_softmax(global_output / 2.0, dim=1), 
                                                F.softmax(smashed_output.detach() / 2.0, dim=1)) * (2.0 * 2.0)
                    global_loss.backward()
                    global_optimizer.step()

                    client_loss.backward(retain_graph=True)
                    grads1 = [param.grad.clone() for param in client_model.parameters()]


                optimizer.zero_grad()
                gradients = u_sr.client(client_socket).to(device)
                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(gradient=smashed_data.grad)

                if args.con_flag == True and round > 0:
                    grads2 = [param.grad.clone() for param in client_model.parameters()]
                    combine_grads = [5 * g1 + g2 for g1, g2 in zip(grads1, grads2)]
                    for param, grad in zip(client_model.parameters(), combine_grads):
                        param.grad = grad
                
                if args.mkd_flag == True and round > 0:
                    grads2 = [param.grad.clone() for param in client_model.parameters()]
                    combine_grads = [g1 + g2 for g1, g2 in zip(grads1, grads2)]
                    for param, grad in zip(client_model.parameters(), combine_grads):
                        param.grad = grad

                if args.moco_flag:
                    grads2 = [param.grad.clone() for param in client_model.parameters()]
                    combine_grads = [5 * g1 + g2 for g1, g2 in zip(grads1, grads2)]
                    for param, grad in zip(client_model.parameters(), combine_grads):
                        param.grad = grad
                
                optimizer.step()
        
        if args.fed_flag == True:
            # 平均化
            client_model = client_model.to('cpu')

            if args.con_flag == True:
                previous_client_model = copy.deepcopy(client_model).to(device)
            
            u_sr.client(client_socket, client_model)
            client_model.load_state_dict(u_sr.client(client_socket).state_dict())
            client_model = client_model.to(device)

            if args.moco_flag:
                for param_q, param_k in zip(client_model.parameters(), global_client_model.parameters()):
                    param_k.data.copy_(param_q.data)
                    param_k.requires_grad = False

            if args.mkd_flag:
                global_client_model = copy.deepcopy(client_model)

            if args.con_flag == True:
                global_client_model = copy.deepcopy(client_model)
                for param in previous_client_model.parameters():
                    param.requires_grad = False
                for param in global_client_model.parameters():
                    param.requires_grad = False


        elif args.fed_flag == False:
            with torch.no_grad():
                for images, labels in tqdm(test_loader):

                    send_data_dict = {'smashed_data': None, 'labels': None}

                    images = images.to(device)
                    labels = labels.to(device)

                    smashed_data = client_model(images)
                    send_data_dict['smashed_data'] = smashed_data
                    send_data_dict['labels'] = labels

                    u_sr.client(client_socket, send_data_dict)
                
                accuracy = u_sr.client(client_socket)
                print('Round: {}, Accuracy: {:.4f}'.format(round+1, accuracy))


    # 通信終了
    client_socket.close()

if __name__ == '__main__':

    args = u_parser.arg_parser()
    print(args)
    main(args)