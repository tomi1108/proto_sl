import sys
sys.path.append('./../')

import argparse
import torch
import torch.nn as nn

import utils.models.mobilenet_v2 as mobilenet_v2


def model_setting(
    args: argparse.ArgumentParser,
    num_classes: int,
    device: str
):
    
    '''
    feature_size: 学習中モデルのProjection headの出力サイズ
    '''
    
    model_name = args.model_name
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    # ココを追加していく
    if model_name == 'mobilenet_v2':

        server_model, client_model, feature_size = mobilenet_v2.create_model(args, num_classes)
    
    else:

        raise ValueError(f'{model_name} is not available.')

    optimizer = torch.optim.SGD(
        params=server_model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    server_model = server_model.to(device)
    client_model = client_model.to(device)

    return server_model, client_model, optimizer, criterion, feature_size