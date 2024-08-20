import torch
from torch.utils.data import DataLoader
from nets.attention_model import set_decode_type


def get_hard_samples(model, data, eps=5, batch_size=1024, device='cpu', baseline=None, get_easy=False):
    """
    Hardness adaptive curriculum, from Zhang et al. (2022)
    https://arxiv.org/abs/2204.03236
    """
    model.eval()
    set_decode_type(model, "greedy")
    multiplier = -1 if get_easy else 1

    # Minmax util function
    def minmax(graphs):
        graphs = (graphs - graphs.min(dim=1,keepdims=True)[0]) \
            / (graphs.max(dim=1,keepdims=True)[0] - graphs.min(dim=1,keepdims=True)[0])
        return graphs

    # Gradient ascent util function
    def get_hard(model, data, eps):
        data = data.to(device)
        data.requires_grad_()
        cost, ll, pi = model(data, return_pi=True)
        if baseline is not None and hasattr(baseline, 'model'):
            with torch.no_grad():
                cost_b, _ = baseline.model(data)
            cost, ll = model(data)
            delta = torch.autograd.grad(eps * ((cost/cost_b) * ll).mean(), data)[0]
        else:
            # As dividend is viewed as constant, it can be omitted in gradient calculation. 
            delta = torch.autograd.grad(eps * (cost*ll).mean(), data)[0]
        ndata = data + (multiplier * delta)
        ndata = minmax(ndata)
        return ndata.detach().cpu()
    
    # Compute hard samples
    batch_size = min(batch_size, len(data))
    dataloader = DataLoader(data, batch_size=batch_size)
    hard = torch.cat([get_hard(model, data, eps) for data in dataloader], dim=0)
    return hard
