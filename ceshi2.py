import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = x
# 反向传播
y.backward()
x.grad

z.backward()
x.grad
