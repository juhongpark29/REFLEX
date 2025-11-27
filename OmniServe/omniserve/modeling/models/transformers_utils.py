import torch
import torch.nn.functional as F
from torch import nn

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def per_token_activation_quantization(hidden_states, verbose=False):
    max_val = hidden_states.abs().max(dim=-1, keepdim=True).values
    max_val = max_val.clamp(min=1e-6)
    scale =  max_val / 127.0
    quantized_hidden_states = (hidden_states / scale).round().to(torch.int8)
    if verbose:
        print(quantized_hidden_states)
        print(scale.squeeze())
    hidden_states = quantized_hidden_states * scale

    return hidden_states, quantized_hidden_states, scale