import sys
sys.path.append('./../../')

import argparse
import csv
import random
import socket
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm

import utils.u_argparser as u_parser
import utils.u_send_receive as u_sr
import utils.u_data_setting as u_dset
import utils.u_print_setting as u_print

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#########################################################################################################
# シード値固定
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
# クライアント通信
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
#########################################################################################################

def main(args: argparse.ArgumentParser):

    set_seed(args.seed)

    if args.save_data:
        dict_path, loss_file_name, accuracy_file_name = u_print.fl_print_setting(args)
        u_print.create_directory(args.results_path + dict_path)
        loss_file_path, accuracy_file_path = u_print.fl_set_output_file(args, dict_path, loss_file_name, accuracy_file_name)
        for connection in connections.values():
            u_sr.server(connection, b"SEND", loss_file_path)
        

    print("========== Server ==========")
    server_socket, connections = set_connection(args)

    test_loader, num_class = u_dset.fl_server_setting(args, connections)
    if args.model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(num_classes=num_class)
        model.classifier[1] = nn.Linear(1280, 64)
        model.classifier.append(nn.ReLU6(inplace=True))
        model.classifier.append(nn.Linear(64, num_class))

    for connection in connections.values():
        u_sr.server(connection, b"SEND", model)
    
    for round in range(args.num_rounds):

        # model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        model_dict = {}
        for client_id, connection in connections.items():
            model_dict[client_id] = u_sr.server(connection, b"REQUEST").to(device)

        model = model.to(device)
        for idx in range(len(model_dict)):
            for param1, param2 in zip(model.parameters(), model_dict[f'Client {idx+1}'].parameters()):
                if idx == 0:
                    param1.data.copy_(param2.data)
                elif 0 < idx < len(model_dict)-1:
                    param1.data.add_(param2)
                else:
                    param1.data.add_(param2.data)
                    param1.data.div_(len(model_dict))
        
        # model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print("Round: {}, Accuracy: {:.2f}%".format(round+1, accuracy))
        
        if args.save_data:
            with open(accuracy_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([round+1, accuracy])
        
        for connection in connections.values():
            u_sr.server(connection, b"SEND", model.to('cpu'))
    
    for client_id, connection in connections.items():
        print("Disconnect from Client {}".format(client_id))
        connection.close()

    server_socket.close()

if __name__ == '__main__':

    args = u_parser.fl_arg_parser()
    print(args)
    main(args)
