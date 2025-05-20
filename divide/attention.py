import torch
import torch.nn as nn

class CustomAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_q = nn.Conv2d(channels, channels, 1)
        self.conv_k = nn.Conv2d(channels, channels, 1)
        self.conv_v = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v)
        return output


