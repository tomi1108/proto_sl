import argparse
import torch
import torch.nn as nn
import torchvision.models as models

class Server_Model(nn.Module):
    def __init__(self, args, model):
        super(Server_Model, self).__init__()

        if args.model_name == 'mobilenet_v2':
            self.model = nn.Sequential(
                model.features[6:],
                nn.Flatten(),
                model.classifier
            )
        else:
            raise Exception(f"The model {args.model_name} is not covered")
    
    def forward(self, x):
        output = self.model(x)
        return output

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