import numpy 
import torch
from torch import nn

# autograd
x = torch.tensor(1.0, requires_grad=True)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

y = w * x + b
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)

# autograd more
x = torch.randn(10, 3)
y = torch.randn(10, 2)
