import argparse
import copy
import torch
import torch.nn.functional as F

class KD:
    def __init__(self,
                 args: argparse.ArgumentParser,
                 client_model: torch.nn.Sequential,
                 projection_head: torch.nn.Sequential,
                 device: torch.device
                 ):
        
        self.args = args
        self.t = args.kd_temperature
        self.device = device
        self.glob_cm = copy.deepcopy(client_model)
        self.proj_head = projection_head
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    def requires_grad_false(self):
        for param in self.glob_cm.parameters():
            param.requires_grad = False

    def train(self,
              images: torch.Tensor,
              smashed_data: torch.Tensor,
              client_model: torch.nn.Sequential,
              optimizer: torch.optim
              ):
        
        optimizer.zero_grad()

        smashed_out = self.proj_head(smashed_data)
        glob_out = self.proj_head(self.glob_cm(images))

        loss = self.criterion(F.log_softmax(smashed_out / self.t, dim=1),
                                     F.softmax(glob_out.detach() / self.t, dim=1)) * (self.t * self.t)      
        loss.backward(retain_graph=True)
        grads = [ param.grad.clone() for param in client_model.parameters() ]

        return grads 
