import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from copy import deepcopy


def variable(t: torch.Tensor, device, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model, bl_cost, dataset, opts):

        self.model = model
        self.bl_cost = bl_cost
        self.dataset = dataset
        self.device = opts.device
        self.ewc_lambda = opts.ewc_lambda

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data, device=self.device)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data, device=self.device)

        self.model.eval()
        self.model.zero_grad()
        cost, log_likelihood = self.model(self.dataset)
        loss = ((cost - self.bl_cost) * log_likelihood).mean()
        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss * self.ewc_lambda
