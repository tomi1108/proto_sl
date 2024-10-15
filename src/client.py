import sys
sys.path.append('./../')

import torch
import torch.nn.functional as F
import socket
import copy
import numpy as np
import argparse
from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_momentum_contrast as u_moco
import utils.u_knowledge_distillation as u_kd
import utils.u_mutual_knowledge_distillation as u_mkd
import utils.u_tiny_moon as u_tim
import utils.u_mixup as u_mix

# 共通モジュールのインポート
import utils.common.u_set_seed as u_seed
# クライアント用モジュールのインポート
import utils.client.uc_data_setting as uc_dset
import utils.client.uc_model_setting as uc_mset
import utils.client.uc_send_receive as uc_sr
# アプローチ用モジュールのインポート
import utils.approach.ua_model_contrastive as ua_moon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# クライアントと通信 
def set_connection(args: argparse.ArgumentParser):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', args.port_number)
    client_socket.connect(server_address)
    client_id = uc_sr.receive(client_socket)
    print("\n========== Client {} ==========\n".format(client_id))

    return client_socket

# main関数 
def main(args: argparse.ArgumentParser):

    # シードを設定
    # set_seed(args.seed)
    u_seed.set_seed(args.seed, device)
    
    # 通信開始
    client_socket = set_connection(args)

    # データローダの作成（通信あり）
    # train_loader, test_loader = uc_dset.data_setting(args, client_socket)
    train_loader, test_loader, data_distribution = uc_dset.set_data_dist(args, client_socket)

    # クライアントモデルを受信
    client_model, optimizer = uc_mset.client_setting(args, client_socket, device)

    if args.save_data:
        # データ分布を送信
        uc_sr.send(client_socket, data_distribution)

    # プロジェクションヘッドの受信
    if args.ph_flag:
        projection_head = uc_sr.receive(client_socket).to(device)

    grads1 = None

    # アプローチごとのインスタンス化を行う
    if args.moco_flag: # Momentum Contrastive Learning
        moco = u_moco.Moco(args, client_model, projection_head, device)
    if args.con_flag: # Model Contrastive Learning
        moon = ua_moon.MOON(args, projection_head, device)
    if args.kd_flag: # Knowledge Distillation
        kd = u_kd.KD(args, client_model, projection_head, device)
    if args.mkd_flag: # Mutual Knowledge Distillation
        mkd = u_mkd.MKD(args, client_model, projection_head, device)
    if args.TiM_flag: # Tiny-MOON
        tim = u_tim.Tim(args, device, projection_head, train_loader)
    if args.Mix_flag:
        mix = u_mix.Mixup_client(args, device)
    

    num_rounds = args.num_rounds
    num_clients = args.num_clients
    fed_flag = args.fed_flag
    moon_flag = args.con_flag

    for round in range(num_rounds):

        client_model.train()
        in_training = True
        print("\n--- Round {}/{} ---".format(round+1, num_rounds))

        while in_training:

            for images, labels in tqdm(train_loader):

                send_data_dict = {'smashed_data': None, 'labels': None}
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                smashed_data = client_model(images)

                send_data_dict['smashed_data'] = smashed_data.to('cpu')
                send_data_dict['labels'] = labels.to('cpu')
                uc_sr.send(client_socket, send_data_dict)

                # 勾配と残り回数を取得する
                receive_data_dict = uc_sr.receive(client_socket)
                gradients = receive_data_dict['gradients'].to(device)
                reminder = receive_data_dict['reminder']

                smashed_data.grad = gradients.clone().detach()
                smashed_data.backward(retain_graph=True, gradient=smashed_data.grad)

                if round > 0:
                    if moon_flag:
                        moon.train(images, smashed_data, client_model, optimizer)

                optimizer.step()

                if reminder < num_clients:
                    in_training = False
                    break
            
        if not in_training:

            if fed_flag:

                if moon_flag:
                    moon.previous_client_model = copy.deepcopy(client_model)

                client_model = client_model.to('cpu')
                client_model.eval()
                uc_sr.send(client_socket, client_model)
                client_model.load_state_dict(uc_sr.receive(client_socket).state_dict())
                client_model = client_model.to(device)

                if moon_flag:
                    moon.global_client_model = copy.deepcopy(client_model)
                    moon.requires_grad_false()
                    if round > 0:
                        moon.reset_loss(round)
            
            else:
                
                # 各クライアントで評価を行うコードを実装
                pass

    # 通信終了
    client_socket.close()


if __name__ == '__main__':

    args = u_parser.arg_parser()
    print(args)
    main(args)


    # # 学習開始
    # for round in range(args.num_rounds):

    #     client_model.train()
    #     print("--- Round {}/{} ---".format(round+1, args.num_rounds))

    #     for epoch in range(args.num_epochs):

    #         print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

    #         for images, labels in tqdm(train_loader):

    #             send_data_dict = {'smashed_data': None, 'labels': None}

    #             if args.moco_flag:
    #                 images, images_k = images[0].to(device), images[1].to(device)
    #             else:
    #                 images = images.to(device)

    #             labels = labels.to(device)
    #             smashed_data = client_model(images)
                
    #             optimizer.zero_grad()

    #             if args.Mix_flag and round > 0:
    #                 smashed_data, labels = mix.mix_smashed_data(images, labels, smashed_data)

    #             send_data_dict['smashed_data'] = smashed_data.to('cpu')
    #             send_data_dict['labels'] = labels.to('cpu')
    #             # u_sr.client(client_socket, send_data_dict)
    #             uc_sr.send(client_socket, send_data_dict)

    #             if args.moco_flag:
    #                 grads1 = moco.train(images, images_k, client_model, optimizer) # モメンタムエンコーダによる勾配を取得

    #             if round > 0:
    #                 if args.con_flag:
    #                     grads1 = moon.train(images, smashed_data.to(device), client_model, optimizer) # モデル対照学習による勾配を取得
    #                 if args.kd_flag:
    #                     grads1 = kd.train(images, smashed_data.to(device), client_model, optimizer)
    #                 if args.mkd_flag:
    #                     grads1 = mkd.train(images, smashed_data.to(device), client_model, optimizer) # 双方向知識蒸留による勾配を取得
    #                 if args.TiM_flag:
    #                     grads1 = tim.train(smashed_data.to(device), labels, client_model, optimizer)

    #             # gradients = u_sr.client(client_socket).to(device)
    #             gradients = uc_sr.receive(client_socket).to(device)
    #             smashed_data.grad = gradients.clone().detach()
    #             smashed_data.backward(gradient=smashed_data.grad)

    #             if grads1 is not None:
    #                 grads2 = [param.grad.clone() for param in client_model.parameters()]
    #                 combine_grads = [5 * g1 + g2 for g1, g2 in zip(grads1, grads2)] # MOON：5, MKD：1, Moco：5, Tim：1
    #                 for param, grad in zip(client_model.parameters(), combine_grads):
    #                     param.grad = grad
                
    #             optimizer.step()
        
    #     if args.fed_flag == True:
            
    #         if args.TiM_flag:
    #             tim.calculate_average(client_model, prev_flag=True)
    #         if args.con_flag:
    #             moon.previous_client_model = copy.deepcopy(client_model)
            
    #         # Aggregation
    #         client_model = client_model.to('cpu')
    #         client_model.eval()
    #         # u_sr.client(client_socket, client_model.to('cpu'))
    #         uc_sr.send(client_socket, client_model.to('cpu'))
    #         # client_model.load_state_dict(u_sr.client(client_socket).state_dict())
    #         client_model.load_state_dict(uc_sr.receive(client_socket).state_dict())
    #         client_model = client_model.to(device)


    #         if args.moco_flag:
    #             moco.global_client_model = copy.deepcopy(client_model)
    #             moco.requires_grad_false()
    #         if args.con_flag:
    #             moon.global_client_model = copy.deepcopy(client_model)
    #             moon.requires_grad_false()
    #         if args.kd_flag:
    #             kd.glob_cm = copy.deepcopy(client_model)
    #             kd.requires_grad_false()
    #         if args.mkd_flag:
    #             mkd.global_client_model = copy.deepcopy(client_model)
    #         if args.TiM_flag:
    #             tim.calculate_average(client_model, glob_flag=True)
    #         if args.Mix_flag:
    #             mix.global_model = copy.deepcopy(client_model)
    #             mix.requires_grad_false()
            


    #     elif args.fed_flag == False:
    #         with torch.no_grad():
    #             for images, labels in tqdm(test_loader):

    #                 send_data_dict = {'smashed_data': None, 'labels': None}

    #                 images = images.to(device)
    #                 labels = labels.to(device)

    #                 smashed_data = client_model(images)
    #                 send_data_dict['smashed_data'] = smashed_data
    #                 send_data_dict['labels'] = labels

    #                 # u_sr.client(client_socket, send_data_dict)
    #                 uc_sr.send(client_socket, send_data_dict)
                
    #             # accuracy = u_sr.client(client_socket)
    #             accuracy = uc_sr.receive(client_socket)
    #             print('Round: {}, Accuracy: {:.4f}'.format(round+1, accuracy))

