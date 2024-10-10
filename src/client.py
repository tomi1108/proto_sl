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
import socket
import numpy as np
import argparse
from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_send_receive as u_sr
import utils.u_data_setting as u_dset
import utils.u_model_setting as u_mset
import utils.u_model_contrast as u_moon
import utils.u_momentum_contrast as u_moco
import utils.u_knowledge_distillation as u_kd
import utils.u_mutual_knowledge_distillation as u_mkd
import utils.u_tiny_moon as u_tim
import utils.u_mixup as u_mix

# 初期設定 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# シードを固定 
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# クライアントと通信 
def set_connection(args: argparse.ArgumentParser):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', args.port_number)
    client_socket.connect(server_address)
    client_id = u_sr.client(client_socket)
    print("\n========== Client {} ==========\n".format(client_id))

    return client_socket

# main関数 
def main(args: argparse.ArgumentParser):

    # シードを設定
    set_seed(args.seed)
    
    # 通信開始
    client_socket = set_connection(args)

    # データローダの作成
    train_loader, test_loader = u_dset.client_data_setting(args, client_socket)

    # クライアントモデルを受信
    client_model, optimizer = u_mset.client_setting(args, client_socket, device)

    # プロジェクションヘッドの受信
    if args.ph_flag:
        projection_head = u_sr.client(client_socket).to(device)

    grads1 = None

    # アプローチごとのインスタンス化を行う
    if args.moco_flag: # Momentum Contrastive Learning
        moco = u_moco.Moco(args, client_model, projection_head, device)
    if args.con_flag: # Model Contrastive Learning
        moon = u_moon.MOON(args, projection_head, device)
    if args.kd_flag: # Knowledge Distillation
        kd = u_kd.KD(args, client_model, projection_head, device)
    if args.mkd_flag: # Mutual Knowledge Distillation
        mkd = u_mkd.MKD(args, client_model, projection_head, device)
    if args.TiM_flag: # Tiny-MOON
        tim = u_tim.Tim(args, device, projection_head, train_loader)
    if args.Mix_flag:
        mix = u_mix.Mixup_client(args, device)

    # 学習開始
    for round in range(args.num_rounds):

        client_model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):

            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for images, labels in tqdm(train_loader):

                send_data_dict = {'smashed_data': None, 'labels': None}

                if args.moco_flag:
                    images, images_k = images[0].to(device), images[1].to(device)
                else:
                    images = images.to(device)

                labels = labels.to(device)
                smashed_data = client_model(images)
                
                optimizer.zero_grad()

                if args.Mix_flag and round > 0:
                    smashed_data, labels = mix.mix_smashed_data(images, labels, smashed_data)

                send_data_dict['smashed_data'] = smashed_data.to('cpu')
                send_data_dict['labels'] = labels.to('cpu')
                u_sr.client(client_socket, send_data_dict)

                if args.moco_flag:
                    grads1 = moco.train(images, images_k, client_model, optimizer) # モメンタムエンコーダによる勾配を取得

                if round > 0:
                    if args.con_flag:
                        grads1 = moon.train(images, smashed_data.to(device), client_model, optimizer) # モデル対照学習による勾配を取得
                    if args.kd_flag:
                        grads1 = kd.train(images, smashed_data.to(device), client_model, optimizer)
                    if args.mkd_flag:
                        grads1 = mkd.train(images, smashed_data.to(device), client_model, optimizer) # 双方向知識蒸留による勾配を取得
                    if args.TiM_flag:
                        grads1 = tim.train(smashed_data.to(device), labels, client_model, optimizer)

                gradients = u_sr.client(client_socket).to(device)
                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(gradient=smashed_data.grad)

                if grads1 is not None:
                    grads2 = [param.grad.clone() for param in client_model.parameters()]
                    combine_grads = [5 * g1 + g2 for g1, g2 in zip(grads1, grads2)] # MOON：5, MKD：1, Moco：5, Tim：1
                    for param, grad in zip(client_model.parameters(), combine_grads):
                        param.grad = grad
                
                optimizer.step()
        
        if args.fed_flag == True:
            
            if args.TiM_flag:
                tim.calculate_average(client_model, prev_flag=True)
            if args.con_flag:
                moon.previous_client_model = copy.deepcopy(client_model)
            
            # Aggregation
            client_model = client_model.to('cpu')
            client_model.eval()
            u_sr.client(client_socket, client_model.to('cpu'))
            client_model.load_state_dict(u_sr.client(client_socket).state_dict())
            client_model = client_model.to(device)


            if args.moco_flag:
                moco.global_client_model = copy.deepcopy(client_model)
                moco.requires_grad_false()
            if args.con_flag:
                moon.global_client_model = copy.deepcopy(client_model)
                moon.requires_grad_false()
            if args.kd_flag:
                kd.glob_cm = copy.deepcopy(client_model)
                kd.requires_grad_false()
            if args.mkd_flag:
                mkd.global_client_model = copy.deepcopy(client_model)
            if args.TiM_flag:
                tim.calculate_average(client_model, glob_flag=True)
            if args.Mix_flag:
                mix.global_model = copy.deepcopy(client_model)
                mix.requires_grad_false()
            


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