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
import utils.u_mutual_knowledge_distillation as u_mkd
import utils.u_tiny_moon as u_tim

#########################################################################################################
# 初期設定 ###############################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pid = os.getpid()
print(f'The process ID (PID) is: {pid}')
time.sleep(5)
#########################################################################################################
# シードを固定 ###########################################################################################
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
#########################################################################################################
# クライアントと通信 ######################################################################################
def set_connection(args: argparse.ArgumentParser):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', args.port_number)
    client_socket.connect(server_address)
    client_id = u_sr.client(client_socket)
    print("\n========== Client {} ==========\n".format(client_id))

    return client_socket
#########################################################################################################
# main関数 ##############################################################################################
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
    if args.con_flag or args.mkd_flag or args.moco_flag or args.Tiny_M_flag:
        projection_head = u_sr.client(client_socket).to(device)

    grads1 = None

    # MOON用のモデルを定義
    if args.con_flag:
        moon = u_moon.MOON(args, projection_head, device)

    # 双方向知識蒸留用モデルを定義
    if args.mkd_flag:
        mkd = u_mkd.MKD(args, client_model, projection_head, device)
    
    if args.moco_flag:
        moco = u_moco.Moco(args, client_model, projection_head, device)

    # Tiny-MOONを使用
    if args.Tiny_M_flag:
        projection_head = u_sr.client(client_socket).to(device)
        criterion = nn.CrossEntropyLoss()
        cos = nn.CosineSimilarity(dim=1)

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
                    grads1 = moco.train(images_q, images_k, client_model, optimizer) # モメンタムエンコーダによる勾配を取得

                if round > 0:
                    if args.con_flag:
                        grads1 = moon.train(images, smashed_data.to(device), client_model, optimizer) # モデル対照学習による勾配を取得
                    if args.mkd_flag:
                        grads1 = mkd.train(images, smashed_data.to(device), client_model, optimizer) # 双方向知識蒸留による勾配を取得

                if args.Tiny_M_flag == True and round > 0:

                    optimizer.zero_grad()                    
                    logits, target = u_tim.calculate_logits(args, smashed_data, labels, global_average, previous_average, projection_head, cos, device)
                    Tiny_M_loss = criterion(logits, target)
                    Tiny_M_loss.backward(retain_graph=True)
                    grads1 = [param.grad.clone() for param in client_model.parameters()]

                optimizer.zero_grad()
                gradients = u_sr.client(client_socket).to(device)
                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(gradient=smashed_data.grad)

                if grads1 is not None:
                    grads2 = [param.grad.clone() for param in client_model.parameters()]
                    combine_grads = [5 * g1 + g2 for g1, g2 in zip(grads1, grads2)] # ここのハイパーパラメータについては後々
                    for param, grad in zip(client_model.parameters(), combine_grads):
                        param.grad = grad

                # MOON：5, MKD：1, Moco：5, Tim：1
                
                optimizer.step()
        
        if args.fed_flag == True:
            
            if args.Tiny_M_flag:
                # 前回ラウンドモデルの出力の平均値
                previous_average = u_tim.calculate_average(args, client_model, projection_head, train_loader)

            # 平均化
            client_model = client_model.to('cpu')

            if args.con_flag:
                moon.previous_client_model = copy.deepcopy(client_model).to(device)
            
            u_sr.client(client_socket, client_model)
            client_model.load_state_dict(u_sr.client(client_socket).state_dict())
            client_model = client_model.to(device)

            if args.moco_flag:
                moco.global_client_model = copy.deepcopy(client_model)
                moco.requires_grad_false()

            if args.con_flag:
                moon.global_client_model = copy.deepcopy(client_model)
                moon.requires_grad_false()

            if args.mkd_flag:
                mkd.global_client_model = copy.deepcopy(client_model)
                
            if args.Tiny_M_flag:
                # 前回ラウンドモデルの出力の平均値
                global_average = u_tim.calculate_average(args, client_model, projection_head, train_loader)


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