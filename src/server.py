import sys
sys.path.append('./../')

import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import socket
import argparse
import os
from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_send_receive as u_sr
import utils.u_data_setting as u_dset
import utils.u_print_setting as u_print
import utils.u_model_setting as u_mset
import utils.u_prototype as u_proto
import utils.u_mixup as u_mix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_model(cfg, num_classes):
    model = models.mobilenet_v2(num_classes=num_classes)
    model.classifier[1] = nn.Linear(1280, cfg['output_size'])
    model.classifier.append(nn.ReLU6(inplace=True))
    model.classifier.append(nn.Linear(cfg['output_size'], num_classes))
    client_model = model.features[:6]
    return model, client_model.to(device)

class Server_Model(nn.Module):
    def __init__(self, model):
        super(Server_Model, self).__init__()

        self.model = nn.Sequential(
            model.features[6:],
            nn.Flatten(),
            model.classifier
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

def set_server_model(model, cfg):
    top_model = Server_Model(model=model)
    optimizer = torch.optim.SGD(params = top_model.parameters(),
                                lr = cfg['lr'],
                                momentum = cfg['momentum'],
                                weight_decay = cfg['weight_decay']
                                )
    criterion = nn.CrossEntropyLoss()
    return top_model, optimizer, criterion

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

def set_output_file(args: argparse.ArgumentParser, dict_path: str, loss_file: str, accuracy_file: str):

    path_to_loss_file = args.results_path + dict_path + '/loss/' + loss_file
    path_to_accuracy_file = args.results_path + dict_path + '/accuracy/' + accuracy_file

    header = ['Round']
    if args.fed_flag == True: # 平均化したモデルでAccuracyを測定する場合
        header += ['Accuracy']
    elif args.fed_flag == False: # 各クライアントモデルでAccuracyを測定する場合
        header += ['Average_Accuracy']
        header += [f'Client{i+1}' for i in range(args.num_clients)]

    with open(path_to_loss_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        if args.proto_flag:
            writer.writerow(['Epoch', 'Loss', 'Proto Loss', 'Total Loss'])
        else:
            writer.writerow(['Epoch', 'Loss'])

    with open(path_to_accuracy_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    return path_to_loss_file, path_to_accuracy_file

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
        u_sr.server(connection, b'SEND', client_id)
        connections['Client {}'.format(client_id)] = connection

    return server_socket, connections    


def main(args: argparse.ArgumentParser):

    # シードを設定して再現性を持たせる
    set_seed(args.seed)

    if args.save_data == True:
        # 初期設定を出力
        dict_path, loss_file_name, accuracy_file_name = u_print.print_setting(args)
        
        # 結果を格納するディレクトリを作成
        create_directory(args.results_path + dict_path)

        # 結果を出力くするファイルを初期化
        loss_path, accuracy_path = set_output_file(args, dict_path, loss_file_name, accuracy_file_name)

    # クライアントと通信開始
    print("========== Server ==========\n")
    server_socket, connections = set_connection(args)
    
    # データセットを用意（SFLの場合のみ）
    test_loader, num_class, num_iterations, num_test_iterations = u_dset.server_data_setting(args, connections) # fed_flag==FalseならNone

    # モデルを定義
    server_model, client_model, optimizer, criterion = u_mset.model_setting(args, num_class)
    server_model = server_model.to(device)
    client_model = client_model.to(device)

    # プロトタイプを定義
    if args.proto_flag:
        prototype = u_proto.prototype(args, num_class)
    if args.Mix_s_flag:
        mix = u_mix.Mixup_server(args, device)
    
    # クライアントにモデルを送信
    for connection in connections.values():
        u_sr.server(connection, b"SEND", client_model)

    # MOON用の次元削減線形層を定義と送信
    if args.ph_flag:
        projection_head = nn.Sequential(
            nn.Flatten()
            # nn.Linear(512, 64)
        )
        for connection in connections.values():
            u_sr.server(connection, b"SEND", projection_head)

    current_epoch = 0
    total_epoch = args.num_rounds * args.num_epochs
    # 学習開始
    for round in range(args.num_rounds):

        server_model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):
            
            current_epoch += 1
            loss_list = []
            proto_loss_list = []
            total_loss_list = []
            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for i in tqdm(range(num_iterations)):

                # if args.Mix_s_flag and round > 0:
                #     client_data_dict = mix.mix_smashed_data(connections)
                #     for client_id, connection in connections.items():
                #         smashed_data = client_data_dict[client_id]['smashed_data'].to(device)
                #         smashed_data.retain_grad()
                #         labels = client_data_dict[client_id]['labels'].to(device)

                #         optimizer.zero_grad()
                #         _, output = server_model(smashed_data)
                #         loss = criterion(output, labels)
                #         loss.backward(retain_graph=True)
                #         loss_list.append(loss.item())
                #         optimizer.step()

                #         u_sr.server(connection, b"SEND", smashed_data.grad.to('cpu'))

                # else:
                #     for connection in connections.values():

                #         client_data = u_sr.server(connection, b"REQUEST")
                #         smashed_data = client_data['smashed_data'].to(device)
                #         smashed_data.retain_grad()
                #         labels = client_data['labels'].to(device)

                #         if smashed_data == None or labels == None:
                #             raise Exception("client_data has None object: {}".format(client_data))

                #         optimizer.zero_grad()
                #         p_output, output = server_model(smashed_data)
                #         loss = criterion(output, labels)

                #         if args.proto_flag:
                #             proto_loss = prototype.compute_prototypes(p_output, labels, device) # ここでプロトタイプの計算を行っている
                #             if proto_loss:
                #                 proto_loss_list.append(proto_loss.item())
                #             if prototype.use_proto:
                #                 total_loss = loss * (current_epoch / total_epoch) + proto_loss * (1 - current_epoch / total_epoch)
                #                 total_loss_list.append(total_loss.item())
                #                 total_loss.backward()
                #             else:
                #                 loss.backward()
                #         else:
                #             loss.backward()

                #         loss_list.append(loss.item())
                #         optimizer.step()

                #         u_sr.server(connection, b"SEND", smashed_data.grad.to('cpu'))

                for connection in connections.values():

                    client_data = u_sr.server(connection, b"REQUEST")
                    smashed_data = client_data['smashed_data'].to(device)
                    smashed_data.retain_grad()
                    labels = client_data['labels'].to(device)

                    if smashed_data == None or labels == None:
                        raise Exception("client_data has None object: {}".format(client_data))

                    optimizer.zero_grad()
                    p_output, output = server_model(smashed_data)
                    loss = criterion(output, labels)

                    loss.backward(retain_graph=True)
                    smashed_data_grad = smashed_data.grad.clone().detach()
                    
                    grads1 = [ param.grad.clone() for param in server_model.parameters() ]

                    if args.proto_flag:
                        proto_loss, grads2 = prototype.compute_prototypes(p_output, labels, server_model) # ここでプロトタイプの計算を行っている
                        if prototype.use_proto:
                            optimizer.zero_grad()
                            proto_loss_list.append(proto_loss.item())
                            # proto_loss.backward(retain_graph=True)
                            # grads2 = [ param.grad.clone() for param in server_model.parameters() ]
                            combine_grad = [ (current_epoch / total_epoch) * g1 + (1 - current_epoch / total_epoch) * g2 for g1, g2 in zip(grads1, grads2)]
                            for param, grad in zip(server_model.parameters(), combine_grad):
                                param.grad = grad

                    optimizer.step()
                    loss_list.append(loss.item())

                    u_sr.server(connection, b"SEND", smashed_data_grad.to('cpu'))

            mean_loss = np.mean(loss_list)
            if args.proto_flag:
                mean_proto_loss = np.mean(proto_loss_list)
                mean_total_loss = np.mean(total_loss_list)
                print(f'Loss: {mean_loss}, Proto Loss: {mean_proto_loss}, Total Loss: {mean_total_loss}')
                if args.save_data:
                    with open(loss_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_epoch, mean_loss, mean_proto_loss, mean_total_loss])
            else:
                print('Loss: ', mean_loss)
                if args.save_data:
                    with open(loss_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_epoch, mean_loss])

        
        if args.fed_flag == True:
            # 平均化
            client_model_dict = {}
            for client_id, connection in connections.items():
                client_model_dict[client_id] = u_sr.server(connection, b"REQUEST").to(device)
            
            client_model = client_model.to(device)
            for idx in range(len(client_model_dict)):
                for param1, param2 in zip(client_model.parameters(), client_model_dict['Client {}'.format(idx+1)].parameters()):
                    if idx == 0:
                        param1.data.copy_(param2.data)
                    elif 0 < idx < len(client_model_dict)-1:
                        param1.data.add_(param2)
                    else:
                        param1.data.add_(param2.data)
                        param1.data.div_(len(client_model_dict))
            
            # 平均化したモデルでテスト
            server_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    p_outputs, outputs = server_model(client_model(images))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print("Round: {}, Accuracy: {:.2f}%".format(round+1, accuracy))

                if args.save_data == True:
                    with open(accuracy_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([round+1, accuracy])
            
            # 平均化モデルをクライアントに送信
            for connection in connections.values():
                u_sr.server(connection, b"SEND", client_model.to('cpu'))

                
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
                        client_data = u_sr.server(connection, b'REQUEST')
                        smashed_data = client_data['smashed_data'].to(device)
                        labels = client_data['labels'].to(device)
                        p_outputs, output = server_model(smashed_data)
                        _, predicted = torch.max(output, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total * 100
                    accuracy_dict[client_id] = accuracy
                    u_sr.server(connection, b'SEND', accuracy)
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
    
    args = u_parser.arg_parser()
    print(args)
    main(args)
