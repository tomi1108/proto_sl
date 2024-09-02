import argparse
import copy
import torch
import torch.nn.functional as F

class MKD:
    def __init__(self, args: argparse.ArgumentParser, client_model, projection_head, device):

        self.args = args
        self.t = args.mkd_temperature
        self.device = device
        self.global_client_model = copy.deepcopy(client_model)
        self.global_optimizer = torch.optim.SGD(params=self.global_client_model.parameters(),
                                           lr=self.args.lr,
                                           momentum=self.args.momentum,
                                           weight_decay=self.args.weight_decay
                                           )
        self.projection_head = projection_head
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')

    def train(self, images, smashed_data, client_model, optimizer):

        optimizer.zero_grad()
        self.global_optimizer.zero_grad()

        smashed_out = self.projection_head(smashed_data)
        global_out = self.projection_head(self.global_client_model(images))

        client_loss = self.criterion(F.log_softmax(smashed_out / self.t, dim=1),
                                     F.softmax(global_out.detach() / self.t, dim=1)) * (self.t * self.t)
        global_loss = self.criterion(F.log_softmax(global_out / self.t, dim=1),
                                     F.softmax(smashed_out.detach() / self.t, dim=1)) * (self.t * self.t)
        global_loss.backward()
        self.global_optimizer.step()

        client_loss.backward(retain_graph=True)
        grads1 = [ param.grad.clone() for param in client_model.parameters() ]

        return grads1