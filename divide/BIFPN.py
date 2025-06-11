import torch
import torch.nn as nn

# BiFPN
class BiFPN_Feature2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Feature2, self).__init__()
        self.epsilon = 0.0001
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)



class BiFPN_Feature3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Feature3, self).__init__()
        self.epsilon = 0.0001
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)

