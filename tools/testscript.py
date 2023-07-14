import random
import torch
import numpy as np
from torch.autograd import Function

class GRL(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha= torch.tensor(0.1, requires_grad=True)
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

x = torch.tensor([5.], requires_grad=True)
y = torch.tensor([12.,], requires_grad=True)
bia = torch.tensor([15., ], requires_grad=True)

z = torch.tensor([6., ], requires_grad=True)
m = torch.tensor([7., ], requires_grad=True)

for i in range(2):
    print(i)
    a = x * y
    b = a + bia

    det = m * b

    b = GRL.apply(b)
    # z.retain_grad()
    domain = z * b

    loss = det + domain

    loss.backward()
    print(x.grad)

