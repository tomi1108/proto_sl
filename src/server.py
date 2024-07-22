import sys
sys.path.append('./../')

import random
import csv
import torch
import torch.nn as nn
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
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created.")

def set_output_file(args: argparse.ArgumentParser, dict_path: str, loss_file: str, accuracy_file: str):
    path_to_loss_file = args.results_path + dict_path + '/' + loss_file
    path_to_accuracy_file = args.results_path + dict_path + '/' + accuracy_file

    with open(path_to_loss_file, mode='w', newline='') as file:
        pass
    with open(path_to_accuracy_file, mode='w', newline='') as file:
        pass

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

    # クライアントにモデルを送信
    for connection in connections.values():
        u_sr.server(connection, b"SEND", client_model)
    

    # 学習開始
    for round in range(args.num_rounds):

        server_model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):

            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for i in tqdm(range(num_iterations)):

                loss_list = []
                for connection in connections.values():

                    client_data = u_sr.server(connection, b"REQUEST")
                    smashed_data = client_data['smashed_data'].to(device)
                    smashed_data.retain_grad()
                    labels = client_data['labels'].to(device)

                    if smashed_data == None or labels == None:
                        raise Exception("client_data has None object: {}".format(client_data))

                    optimizer.zero_grad()
                    outputs = server_model(smashed_data)
                    loss = criterion(outputs, labels)

                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    u_sr.server(connection, b"SEND", smashed_data.grad.to('cpu'))


                if i % 100 == 0:
                    average_loss = sum(loss_list) / len(loss_list)
                    print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, i+1, average_loss))
                    cur_iter = i + num_iterations * epoch + round * num_iterations * args.num_epochs
                    with open(loss_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([cur_iter, average_loss])
        
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
            client_model.eval()
            correct = 0
            total = 0
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = server_model(client_model(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print("Round: {}, Accuracy: {:.2f}%".format(round+1, accuracy))

            with open(accuracy_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([round+1, accuracy])
            
            # 平均化モデルをクライアントに送信
            for connection in connections.values():
                u_sr.server(connection, b"SEND", client_model.to('cpu'))
        else:
            # 各クライアントでテストを行う
            pass


    for client_id, connection in connections.items():
        print("Disconnect from Client {}".format(client_id))
        connection.close()

    server_socket.close()

if __name__ == '__main__':
    
    args = u_parser.arg_parser()
    print(args)
    main(args)

    # # TRAINING ###########################################################

    # for round in range(config['rounds']):
        
    #     top_model.train()
    #     print("--- Round {}/{} ---".format(round+1, config['rounds']))
        
    #     for epoch in range(config['epochs']):
            
    #         print("--- Epoch {}/{} ---".format(epoch+1, config['epochs']))

    #         for i in tqdm(range(num_iterations)):

    #             loss_list = []
    #             for connection in connection_list:

    #                 client_data = m_sr.server(connection, b"REQUEST")
    #                 smashed_data = client_data['smashed_data'].to(device)
    #                 labels = client_data['labels'].to(device)

    #                 optimizer.zero_grad()
    #                 outputs = top_model(smashed_data)

    #                 loss = criterion(outputs, labels)

    #                 loss_list.append(loss.item())
    #                 loss.backward()
    #                 optimizer.step()
    #                 m_sr.server(connection, b"SEND", smashed_data.grad)
                
    #             if i % 100 == 0:
    #                 average_loss = sum(loss_list) / len(loss_list)
    #                 print("Round: {} | Epoch: {} | Iteration: {} | Loss: {:.4f}".format(round+1, epoch+1, i+1, average_loss))
    #                 cur_iter = i + num_iterations * epoch + round * num_iterations * config['epochs']
    #                 with open(loss_path, 'a') as f:
    #                     f.write("Iteration: {}, Loss: {:.4f}\n".format(cur_iter, average_loss))
        
    #     # AVERAGING ########################################################

    #     client_model_list = []
    #     for connection in connection_list:
    #         client_model_list.append(m_sr.server(connection, b"REQUEST"))

    #     for idx in range(len(client_model_list)):
    #         for param1, param2 in zip(client_model.parameters(), client_model_list[idx].parameters()):
    #             if idx == 0:
    #                 param1.data.copy_(param2.data)
    #             elif 0 < idx < len(client_model_list)-1:
    #                 param1.data.add_(param2.data)
    #             else:
    #                 param1.data.add_(param2.data)
    #                 param1.data.div_(len(client_model_list))

    #     # TEST ############################################################

    #     top_model.eval()
    #     correct = 0
    #     total = 0
    #     for images, labels in tqdm(test_loader):
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = top_model(client_model(images))
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #     accuracy = 100 * correct / total
    #     print("Round: {}, Accuracy: {:.2f}%".format(round+1, accuracy))

    #     with open(accuracy_path, 'a') as f:
    #         f.write("Round: {} : Accuracy: {:.2f}\n".format(round+1, accuracy))
        
    #     # SEND MODEL ######################################################

    #     for connection in connection_list:
    #         m_sr.server(connection, b"SEND", client_model)
    
    # # CLOSE CONNECTION #####################################################

    # for connection in connection_list:
    #     connection.close()
    # server_socket.close()