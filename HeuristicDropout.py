import torch
import torch.nn as nn
import torch.nn.functional as F


class HeuristicDropout(nn.Module):
    def __init__(self, rate=0.1, threshold=0.5):
        super().__init__()
        self.rate = rate
        self.threshold = threshold
        self.bin_count = 10
    
    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            xtype, xdevice = x.dtype, x.device
            x_tanh = torch.tanh(x)
            filter_identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], device=xdevice, dtype=xtype)
            filter_laplace = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device=xdevice, dtype=xtype)
            var = torch.var(x, dim=(2, 3))
            quantize = torch.round(x_tanh * self.bin_count).view(b, c, -1)
            hist = torch.count_nonzero(quantize.unsqueeze(-1) == torch.arange(0, self.bin_count+1, device=xdevice), dim=2)
            entropy = torch.distributions.Categorical(hist).entropy()
            _, indices = torch.sort(entropy+2./(var+1e-7), dim=1, descending=True)

            filters = filter_identity.repeat(b * c, 1, 1, 1)
            indices_ = indices + (torch.arange(0, b, device=xdevice) * c).repeat(c, 1).transpose(1, 0)
            indices_ = torch.flatten(indices_[:, :round(self.rate * c)])
            filters[indices_, 0] = filter_laplace
            x_ = torch.cat(torch.split(x, 1, dim=0), dim=1)
            outx_ = F.conv2d(x_, filters, padding=1, groups=b * c)
            outx = torch.cat(torch.split(outx_, c, dim=1), dim=0)
            
            return outx
        else:
            return x


class AlternativeRound(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class HeuristicDropoutWithAlternativeRound(nn.Module):
    def __init__(self, rate=0.1, threshold=0.5):
        super().__init__()
        self.rate = rate
        self.threshold = threshold
        self.bin_count = 10
    
    def forward(self, x):
        if self.training:
            b, c, h, w = x.size()
            xtype, xdevice = x.dtype, x.device
            x_tanh = torch.tanh(x)
            filter_identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], device=xdevice, dtype=xtype)
            filter_laplace = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], device=xdevice, dtype=xtype)
            var = torch.var(x, dim=(2, 3))
            alternative_round = AlternativeRound.apply
            quantize = alternative_round(x_tanh * self.bin_count).view(b, c, -1)
            hist = torch.count_nonzero(quantize.unsqueeze(-1) == torch.arange(0, self.bin_count+1, device=xdevice), dim=2)
            entropy = torch.distributions.Categorical(hist).entropy()
            _, indices = torch.sort(entropy+2./(var+1e-7), dim=1, descending=True)

            filters = filter_identity.repeat(b * c, 1, 1, 1)
            indices_ = indices + (torch.arange(0, b, device=xdevice) * c).repeat(c, 1).transpose(1, 0)
            indices_ = torch.flatten(indices_[:, :round(self.rate * c)])
            filters[indices_, 0] = filter_laplace
            x_ = torch.cat(torch.split(x, 1, dim=0), dim=1)
            outx_ = F.conv2d(x_, filters, padding=1, groups=b * c)
            outx = torch.cat(torch.split(outx_, c, dim=1), dim=0)
            
            return outx
        else:
            return x
