import argparse
import copy
import torch
import torch.nn.functional as F

class Moco:
    def __init__(self, args: argparse.ArgumentParser, client_model, projection_head, device):

        self.device = device
        self.args = args
        self.global_client_model = copy.deepcopy(client_model)
        for param in self.global_client_model.parameters():
            param.requires_grad = False
        self.projection_head = projection_head
        self.queue_size = self.args.queue_size
        self.m = 0.999
        self.T = args.temperature
        self.queue = torch.randn(args.output_size, self.queue_size).to(self.device)
        self.pointer = torch.zeros(1, dtype=torch.long)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train(self, images_q, images_k, client_model, optimizer):
        
        optimizer.zero_grad()

        q = F.normalize(self.projection_head(client_model(images_q)), dim=1)
        with torch.no_grad():
            for param_q, param_k in zip(client_model.parameters(), self.global_client_model.parameters()):
                param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
            k = F.normalize(self.projection_head(self.global_client_model(images_k)), dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        self.update_queue(k)

        loss = self.criterion(logits, labels)
        loss.backward(retain_graph=True)
        
        grads1 = [ param.grad.clone() for param in client_model.parameters() ]

        return grads1

    def update_queue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.pointer)
        assert self.queue_size % batch_size == 0

        self.queue[:, ptr : ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.queue_size

        self.pointer[0] = ptr

    def requires_grad_false(self):
        for param in self.global_client_model.parameters():
            param.requires_grad = False
        