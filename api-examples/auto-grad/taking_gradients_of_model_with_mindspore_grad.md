# 使用mindspore对模型求导

> 备注：以下示例使用的是 MindSpore 2.4.0 版本

## 计算模型输出对权重的梯度

先定义一个简单的模型

```python
from mindspore import nn, Parameter
import numpy as np

np.random.seed(0)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(np.random.randn(4, 4))
        self.b = Parameter(np.random.randn(4,))

    def construct(self, x, y):
        pred = self.w @ x + self.b
        loss = (pred - y) ** 2
        return loss.sum()

net = Net()
    
print(f"net: \n{net}")
print(f"parameters of net: \n{net.get_parameters()}")
```

使用`mindspore.grad`接口计算模型对权重的梯度

> 注意：这里不需要计算模型对输入的梯度，所以设置`grad_position=None`

```python
from mindspore import grad, Tensor
import numpy as np

x, y = Tensor(np.random.randn(2, 4)), Tensor([1.0])

grad_fn = grad(net, grad_position=None, weights=net.get_parameters())

grads = grad_fn(x, y)

print(f"grads: \n{grads}")
```

当然我们也可以使用`mindspore.value_and_grad`接口计算模型对权重的梯度并获取`loss`的情况

```python
from mindspore import value_and_grad

grad_fn = value_and_grad(net, grad_position=None, weights=net.get_parameters())

loss, grads = grad_fn(x, y)

print(f"loss: {loss}")
print(f"grads: \n{grads}")
```

使用 `JIT(Just-In-Time)` 编译加速

```python
from mindspore import jit

@jit
def loss_and_grads():
    return grad_fn(x, y)

loss, grads = loss_and_grads(x, y)

print(f"loss: {loss}")
print(f"grads: \n{grads}")
```
