import argparse
import torch.nn as nn
import torchvision.models as models

def create_model(
    args: argparse.ArgumentParser,
    num_classes: int
):
    
    model = models.mobilenet_v2(pretrained=False)
    dataset = args.dataset_type
    
    if dataset == 'cifar10':
        
        feature_size = 128

        model.classifier[1] = nn.Linear(1280, feature_size)
        model.classifier.append(nn.ReLU(inplace=False))
        model.classifier.append(nn.Dropout(p=0.2, inplace=False))
        model.classifier.append(nn.Linear(128, num_classes, bias=True))
    
    elif dataset == 'cifar100':

        feature_size = 256

        model.classifier[1] = nn.Linear(1280, feature_size)
        model.classifier.append(nn.ReLU(inplace=False))
        model.classifier.append(nn.Dropout(p=0.2, inplace=False))
        model.classifier.append(nn.Linear(256, num_classes, bias=True))
    
    else:

        raise ValueError(f'{dataset} is not available.')
    
    client_model = model.features[:6]
    server_model = Server_model(model)

    return server_model, client_model, feature_size


class Server_model(nn.Module):

    def __init__(self, model):
        super(Server_model, self).__init__()

        self.base_encoder = nn.Sequential(
            model.features[6:]
        )

        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model.classifier[:-1]
        )

        self.output_layer = model.classifier[-1]
    
    def forward(self, x):
        
        feature_vector = self.base_encoder(x)
        projected_feature = self.projection_head(feature_vector)
        predictions = self.output_layer(projected_feature)

        return projected_feature, predictions