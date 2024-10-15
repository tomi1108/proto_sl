import sys
sys.path.append('./../')

import random
import csv
import torch
import torch.nn as nn
import numpy as np
import socket
import argparse
from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_mixup as u_mix

# 共通モジュールのインポート
import utils.common.u_set_seed as u_seed
import utils.common.u_save_data as u_save
# サーバ用モジュールのインポート
import utils.server.us_print_setting as us_print
import utils.server.us_result_output as us_result
import utils.server.us_data_setting as us_dset
import utils.server.us_model_setting as us_mset
import utils.server.us_send_receive as us_sr
import utils.server.us_model_aggregation as us_magg
import utils.server.us_model_testing as us_test
# アプローチ用モジュールのインポート
import utils.approach.ua_prototype_contrastive as ua_proto


def set_connection(args: argparse.ArgumentParser):
    connection_list = []
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('localhost', args.port_number)
    server_socket.bind(server_address)
    server_socket.listen(args.num_clients)
    print("Waiting for Clients...")

    while len(connection_list) < args.num_clients:
        connection, client_address = server_socket.accept()
        connection_list.append(connection)
        print("Connected to Client: {}".format(client_address))
    
    connections = {}
    for i, connection in enumerate(connection_list):
        client_id = i+1
        us_sr.send(connection, client_id)
        connections['Client {}'.format(client_id)] = connection

    return server_socket, connections    


def main(args: argparse.ArgumentParser):

    # シード設定
    u_seed.set_seed(args.seed, device)

    # 結果を記録するCSVファイルへのパスを設定
    if args.save_data:
        result_dir_path, file_name = us_print.print_setting(args)
        acc_dir_path, dist_dir_path, loss_dir_path = us_result.create_directory(args, result_dir_path)
        accuracy_path, loss_path = us_result.set_output_file(args, acc_dir_path, loss_dir_path, file_name)
        
    # 通信開始
    print("========== Server ==========\n")
    server_socket, connections = set_connection(args)
    
    # テストローダ作成（通信あり）
    test_loader, num_classes, num_train_iterations, num_test_iterations, weighs_in_model_aggregation = us_dset.set_client_data_dist(args, connections)

    # モデルを定義
    server_model, client_model, optimizer, criterion, feature_size = us_mset.model_setting(args, num_classes, device)

    # プロトタイプを定義
    if args.proto_flag:
        # prototype = u_proto.prototype(args, num_classes)
        prototype = ua_proto.prototype(args, num_classes, feature_size, device)
    if args.Mix_s_flag:
        mix = u_mix.Mixup_server(args, device)
    
    # クライアントにモデルを送信
    for connection in connections.values():
        us_sr.send(connection, client_model.to('cpu'))
    
    # 各クライアントのデータ分布を保存
    if args.save_data:
        distribution_dict = {}
        for client_id, connection in connections.items():
            distribution_dict[client_id] = us_sr.receive(connection)
        us_result.save_data_dist(num_classes, dist_dir_path, file_name, distribution_dict)    

    # MOON用の次元削減線形層を定義と送信
    if args.ph_flag:
        projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 64)
        )
        for connection in connections.values():
            us_sr.send(connection, projection_head)

    current_epoch = 0
    num_clients = args.num_clients
    num_rounds = args.num_rounds
    num_epochs = args.num_epochs
    total_epoch = num_rounds * num_epochs
    client_number = 1

    # 学習開始
    for round in range(num_rounds):

        server_model.train()
        print("\n--- Round {}/{} ---".format(round+1, num_rounds))

        for epoch in range(num_epochs):
            
            current_epoch += 1
            mu = current_epoch / total_epoch

            runnning_loss = 0.0
            if args.proto_flag:
                prototype.reset_loss()

            print("--- Epoch {}/{} ---".format(epoch+1, num_epochs))

            for iter in tqdm(range(num_train_iterations)):

                connection = connections['Client {}'.format(client_number)]

                client_data = us_sr.receive(connection)
                smashed_data = client_data['smashed_data'].to(device)
                labels = client_data['labels'].to(device)
                smashed_data.retain_grad()

                if smashed_data == None or labels == None:
                    raise Exception("client_data has None object: {}".format(client_data))

                optimizer.zero_grad()
                p_output, output = server_model(smashed_data)
                loss = criterion(output, labels)
                runnning_loss += loss.item()

                if args.proto_flag:

                    # use_protoがTrueなら、totalのlossに置き換わる                        
                    loss = prototype.compute_prototypes(p_output, labels, loss, mu)

                loss.backward()
                optimizer.step()

                smashed_data_grad = smashed_data.grad.clone().detach()
                send_data_dict = {'gradients': None, 'reminder': None}

                send_data_dict['gradients'] = smashed_data_grad.to('cpu')
                if epoch == num_epochs-1:
                    send_data_dict['reminder'] = num_train_iterations - (iter + 1)
                else:
                    send_data_dict['reminder'] = num_clients

                us_sr.send(connection, send_data_dict)

                # 次に通信するクライアントを指定
                client_number += 1
                if client_number > num_clients:
                    client_number = 1

                # for connection in connections.values():

                #     client_data = us_sr.receive(connection)
                #     smashed_data = client_data['smashed_data'].to(device)
                #     labels = client_data['labels'].to(device)
                #     smashed_data.retain_grad()

                #     if smashed_data == None or labels == None:
                #         raise Exception("client_data has None object: {}".format(client_data))

                #     optimizer.zero_grad()
                #     p_output, output = server_model(smashed_data)
                #     loss = criterion(output, labels)
                #     runnning_loss += loss.item()

                #     if args.proto_flag:

                #         # use_protoがTrueなら、totalのlossに置き換わる                        
                #         loss = prototype.compute_prototypes(p_output, labels, loss, mu)

                #     loss.backward()
                #     optimizer.step()

                #     smashed_data_grad = smashed_data.grad.clone().detach()
                #     send_data_dict = {'gradients': None, 'reminder': None}

                #     send_data_dict['gradients'] = smashed_data_grad.to('cpu')
                #     if epoch == num_epochs-1:
                #         send_data_dict['reminder'] = num_train_iterations - (iter + 1)
                #     else:
                #         send_data_dict['reminder'] = num_clients

                #     # us_sr.send(connection, smashed_data_grad.to('cpu'))
                #     us_sr.send(connection, send_data_dict)
                    
                    # main_grad = [ param.grad.clone() for param in server_model.parameters() ]

                    # if args.proto_flag:
                    #     proto_loss, grads2 = prototype.compute_prototypes(p_output, labels, server_model) # ここでプロトタイプの計算を行っている
                    #     if prototype.use_proto:
                    #         optimizer.zero_grad()
                    #         proto_loss_list.append(proto_loss.item())
                    #         # proto_loss.backward(retain_graph=True)
                    #         # grads2 = [ param.grad.clone() for param in server_model.parameters() ]
                    #         combine_grad = [ (current_epoch / total_epoch) * g1 + (1 - current_epoch / total_epoch) * g2 for g1, g2 in zip(main_grad, grads2)]
                    #         for param, grad in zip(server_model.parameters(), combine_grad):
                    #             param.grad = grad


            mean_loss = runnning_loss / num_train_iterations

            if args.proto_flag:
                
                mean_proto_loss, mean_total_loss = prototype.get_loss()
                print(f'Loss: {mean_loss}, Proto Loss: {mean_proto_loss}, Total Loss: {mean_total_loss}')
                write_data = [current_epoch, mean_loss, mean_proto_loss, mean_total_loss]

            else:

                print('Loss: ', mean_loss)
                write_data = [current_epoch, mean_loss]

            if args.save_data:
                u_save.save_data(loss_path, write_data)

        
        if args.fed_flag:
            
            # クライアントモデルを受信
            client_model_dict = {}
            for client_id, connection in connections.items():
                client_model_dict[client_id] = us_sr.receive(connection).to(device)
            client_model.to(device)

            # グローバルクライアントモデルを更新
            client_model = us_magg.model_aggregation(
                client_model,
                client_model_dict,
                weighs_in_model_aggregation
            )
            
            # グローバルモデルでテスト
            server_model.eval()
            accuracy, test_loss = us_test.test_with_global_model(
                server_model,
                client_model,
                test_loader,
                device
            )
            print(f"Round: {round+1}, Accuracy: {accuracy}, Loss: {test_loss}")

            if args.save_data:
                u_save.save_data(accuracy_path, [round+1, accuracy])
            
            # 平均化モデルをクライアントに送信
            for connection in connections.values():
                us_sr.send(connection, client_model.to('cpu'))

                
        elif args.fed_flag == False:
            # 各クライアントでテストを行う
            server_model.eval()
            with torch.no_grad():
                accuracy_dict = {}
                accuracy_dict['Round'] = round+1
                accuracy_dict['Average_Accuracy'] = None
                for client_id, connection in connections.items():
                    correct = 0
                    total = 0
                    for i in tqdm(range(num_test_iterations)):
                        # client_data = u_sr.server(connection, b'REQUEST')
                        client_data = us_sr.receive(connection)
                        smashed_data = client_data['smashed_data'].to(device)
                        labels = client_data['labels'].to(device)
                        p_outputs, output = server_model(smashed_data)
                        _, predicted = torch.max(output, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total * 100
                    accuracy_dict[client_id] = accuracy
                    # u_sr.server(connection, b'SEND', accuracy)
                    us_sr.send(connection, accuracy)
            print('Round: {}'.format(round+1), end=',')
            accuracy_list = []
            for idx in range(args.num_clients):
                if idx < args.num_clients-1:
                    print('Client{}: {}'.format(idx+1, accuracy_dict['Client {}'.format(idx+1)]), end=', ')
                elif idx == args.num_clients-1:
                    print('Client{}: {}'.format(idx+1, accuracy_dict['Client {}'.format(idx+1)]))
                accuracy_list.append(accuracy_dict['Client {}'.format(idx+1)])
            accuracy_dict['Average_Accuracy'] = sum(accuracy_list) / len(accuracy_list)
            print('Average Accuracy: {}'.format(accuracy_dict['Average_Accuracy']))
            values = accuracy_dict.values()

            if args.save_data == True:
                with open(accuracy_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(values)


    for client_id, connection in connections.items():
        print("Disconnect from Client {}".format(client_id))
        connection.close()

    server_socket.close()

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = u_parser.arg_parser()
    print(args)
    main(args)
