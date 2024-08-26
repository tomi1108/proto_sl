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
# サーバと通信
def set_connection(args: argparse.ArgumentParser):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', args.port_number)
    client_socket.connect(server_address)
    client_id = u_sr.client(client_socket)

    return client_socket, client_id
#########################################################################################################


def main(args: argparse.ArgumentParser):

    set_seed(args.seed)

    client_socket, client_id = set_connection(args)
    print("\n========== Client {} ==========\n".format(client_id))

    if args.save_data:
        loss_file_path = u_sr.client(client_socket)

    train_loader = u_dset.fl_client_setting(args, client_socket)

    model = u_sr.client(client_socket).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    criterion = nn.CrossEntropyLoss()
    
    for round in range(args.num_rounds):

        model.train()
        print("--- Round {}/{} ---".format(round+1, args.num_rounds))

        for epoch in range(args.num_epochs):
            
            loss_list = []
            print("--- Epoch {}/{} ---".format(epoch+1, args.num_epochs))

            for i, (images, labels) in enumerate(tqdm(train_loader)):

                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            
            print("Round: {}, Epoch: {}, Loss: {}".format(round+1, epoch+1, np.mean(loss_list)))
            if args.save_data:
                with open(loss_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, np.mean(loss_list)])
            
        model = model.to('cpu')
        u_sr.client(client_socket, model)
        model.load_state_dict(u_sr.client(client_socket).state_dict())
        model = model.to(device)

if __name__ == '__main__':

    args = u_parser.fl_arg_parser()
    print(args)
    main(args)
