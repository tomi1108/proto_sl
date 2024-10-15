import argparse
import torch
import torch.nn as nn

class MOON:
    def __init__(
        self,
        args: argparse.ArgumentParser,
        projection_head: nn.Sequential,
        device: str
    ):
        
        self.mu = 5
        self.running_loss = 0.0
        self.loss_count = 0
        self.temperature = args.moon_temperature
        self.device = device

        self.previous_client_model = None
        self.global_client_model = None
        self.projection_head = projection_head

        self.criterion = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    
    def train(
        self,
        images: torch.Tensor,
        smashed_data: torch.Tensor,
        client_model: nn.Sequential,
        optimizer: torch.optim
    ):
        
        main_grad = [ param.grad.clone() for param in client_model.parameters() ]
        optimizer.zero_grad()

        smashed_output = self.projection_head(smashed_data)
        previous_output = self.projection_head(self.previous_client_model(images))
        global_output = self.projection_head(self.global_client_model(images))

        pos = self.cosine_similarity(smashed_output, global_output)
        logits = pos.reshape(-1, 1)
        neg = self.cosine_similarity(smashed_output, previous_output)
        logits = torch.cat((logits, neg.reshape(-1, 1)), dim=1)

        logits /= self.temperature
        logits_labels = torch.zeros(images.size(0)).to(self.device).long()

        loss = self.criterion(logits, logits_labels)
        loss.backward()
        self.running_loss += loss.item()
        self.loss_count += 1

        moon_grads = [ param.grad.clone() for param in client_model.parameters() ]
        combine_grad = [ self.mu * g_moon + g_main for g_moon, g_main in zip(moon_grads, main_grad)]
        for param, grad in zip(client_model.parameters(), combine_grad):
            param.grad = grad
        
        return loss.item()


    def requires_grad_false(self):

        for param_p, param_g in zip(self.previous_client_model.parameters(), self.global_client_model.parameters()):
            param_p.requires_grad = False
            param_g.requires_grad = False
    

    def reset_loss(self, round: int):

        print(f"Round: {round}, MOON Loss: {self.running_loss / self.loss_count}")
        self.running_loss = 0.0
        self.loss_count = 0