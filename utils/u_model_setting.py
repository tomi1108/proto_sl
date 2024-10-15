import argparse
import socket
import torch

import utils.u_send_receive as u_sr
import utils.client.uc_send_receive as uc_sr

def client_setting(args: argparse.ArgumentParser, client_socket: socket.socket, device):
    
    # client_model = u_sr.client(client_socket).to(device)
    client_model = uc_sr.receive(client_socket).to(device)
    optimizer = torch.optim.SGD(params=client_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    return client_model, optimizer