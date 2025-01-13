# 创建一个简单的神经网络

```python
from mindspore import nn, Tensor
import numpy as np

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mlp = nn.SequentialCell([
            nn.Dense(16*16, 128),
            nn.ReLU(),
            nn.Dense(128, 128),
            nn.ReLU(),
            nn.Dense(128, 10),
        ])
        self.loss_fn = nn.CrossEntropyLoss()

    def construct(self, x, y):
        x = self.mlp(x)
        loss = self.loss_fn(x, y)
        return loss

net = Net()
x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))

print(net)
print(net(x, y))
```
