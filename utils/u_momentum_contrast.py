import argparse
import torch

class Moco:
    def __init__(self, args: argparse.ArgumentParser):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.queue_size = self.args.queue_size
        self.m = 0.999
        self.T = args.temperature
        self.queue = torch.randn(args.output_size, self.queue_size).to(self.device)
        self.moco_ptr = torch.zeros(1, dtype=torch.long)

    def compute_moco(self, im_q, im_k, q_mdl, k_mdl, ph):

        q = torch.nn.functional.normalize(ph(q_mdl(im_q)), dim=1)
        with torch.no_grad():
            for param_q, param_k in zip(q_mdl.parameters(), k_mdl.parameters()):
                param_k.data = param_k.data * self.m + param_q * (1.0 - self.m)
            k = torch.nn.functional.normalize(ph(k_mdl(im_k)), dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        self.update_queue(k)

        return logits, labels

    def update_queue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.moco_ptr)
        assert self.queue_size % batch_size == 0

        self.queue[:, ptr : ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.queue_size

        self.moco_ptr[0] = ptr


