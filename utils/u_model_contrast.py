import argparse
import torch

class MOON:
    def __init__(self, args:argparse.ArgumentParser, projection_head, device):
        
        self.args = args
        self.device = device
        self.previous_client_model = None
        self.global_client_model = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.projection_head = projection_head
    
    def requires_grad_false(self):
        for param_p, param_g in zip(self.previous_client_model.parameters(), self.global_client_model.parameters()):
            param_p.requires_grad = False
            param_g.requires_grad = False
    
    def train(self, images, smashed_data, client_model, optimizer):

        optimizer.zero_grad()

        smashed_out = self.projection_head(smashed_data)
        previous_out = self.projection_head(self.previous_client_model(images))
        global_out = self.projection_head(self.global_client_model(images))

        pos = self.cos(smashed_out, global_out)
        logits = pos.reshape(-1, 1)
        neg = self.cos(smashed_out, previous_out)
        logits = torch.cat((logits, neg.reshape(-1, 1)), dim=1)

        logits /= self.args.moon_temperature
        logits_labels = torch.zeros(images.size(0)).to(self.device).long()

        con_loss = self.criterion(logits, logits_labels)
        con_loss.backward(retain_graph=True)
        grads1 = [ param.grad.clone() for param in client_model.parameters() ]

        return grads1