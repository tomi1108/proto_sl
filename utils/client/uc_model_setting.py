import argparse
import socket
import torch

import utils.client.uc_send_receive as uc_sr

def client_setting(
    args: argparse.ArgumentParser,
    client_socket: socket.socket,
    device: str
):
    
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    
    client_model = uc_sr.receive(client_socket).to(device)
    optimizer = torch.optim.SGD(
        params=client_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    return client_model, optimizer