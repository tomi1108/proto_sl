import argparse
import socket
import torch
import torch.nn as nn
import torchvision.models as models
import utils.u_send_receive as u_sr

class Server_Model(nn.Module):
    def __init__(self, args, model):
        super(Server_Model, self).__init__()

        self.args = args

        if args.model_name == 'mobilenet_v2':
            self.model1 = nn.Sequential(
                model.features[6:],
                nn.Flatten(),
                model.classifier[0],
                model.classifier[1]
            )
            self.model2 = nn.Sequential(
                model.classifier[2],
                model.classifier[3]
            )
        else:
            raise Exception(f"The model {args.model_name} is not covered")
    
    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(output1)
        return output1, output2

def model_setting(args: argparse.ArgumentParser, num_class: int):

    if args.model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(num_classes=num_class)
        model.classifier[1] = nn.Linear(1280, args.output_size)
        model.classifier.append(nn.ReLU6(inplace=True))
        model.classifier.append(nn.Linear(args.output_size, num_class))
        client_model = model.features[:6]
        server_model = Server_Model(args, model)
    else:
        raise Exception(f"The model {args.model_name} is not covered")
    
    optimizer = torch.optim.SGD(params=server_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    return server_model, client_model, optimizer, criterion

def client_setting(args: argparse.ArgumentParser, client_socket: socket.socket, device):
    
    client_model = u_sr.client(client_socket).to(device)
    optimizer = torch.optim.SGD(params=client_model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )
    return client_model, optimizer