import argparse
import torch

class prototype:
    def __init__(self, args: argparse.ArgumentParser, num_class: int):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.num_class = num_class
        self.queue_size = args.queue_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.queue_features = torch.zeros(self.queue_size, args.output_size)
        self.queue_labels = torch.zeros(self.queue_size, dtype=torch.long)
        self.prototypes = torch.zeros(num_class, args.output_size)
        self.proto_pointer = torch.zeros(1, dtype=torch.long)
        self.use_proto = False
        self.cos = torch.nn.CosineSimilarity(dim=1)
    
    def compute_prototypes(self, output: torch.Tensor, labels: torch.Tensor, device: str):
        
        with torch.no_grad():
            ptr = int(self.proto_pointer)
            self.queue_features[ptr:ptr + self.args.batch_size] = output
            self.queue_labels[ptr:ptr + self.args.batch_size] = labels
            ptr = (ptr + self.args.batch_size) % self.args.queue_size
            self.proto_pointer[0] = ptr

            if ptr == 0:
                self.use_proto = True
                for idx in range(self.num_class):
                    self.prototypes[idx] = self.queue_features[self.queue_labels == idx].mean(0)
                self.prototypes = torch.nn.functional.normalize(self.prototypes, dim=1)
        
        if self.use_proto:
            output = torch.nn.functional.normalize(output, dim=1)

            # similarities = torch.mm(output, self.prototypes.t().to(device)).to(device)

            # exp_similarities = torch.exp(similarities)
            # denominator = exp_similarities.sum(dim=1, keepdim=True)

            # proto_loss = -torch.log(exp_similarities[range(output.size(0)), labels].unsqueeze(1) / denominator).mean()


            # similarities = torch.zeros(output.size(0), self.prototypes.size(0)).to(device)
            # for i, proto in enumerate(self.prototypes.to(device)):
            #     similarities[:, i] = self.cos(output, proto.unsqueeze(0).expand_as(output))
            # proto_loss = self.criterion(similarities, labels)

            proto_o = torch.mm(output, self.prototypes.t().to(self.device))
            proto_o /= self.args.temperature
            proto_loss = self.criterion(proto_o, labels)

            return proto_loss
        else:
            return None