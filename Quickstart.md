## Quickstart

本节介绍了如何使用 MindSpore 进行简单的模型训练。


### 1. 创建模型

```python
from mindspore import nn

class MLP(nn.Cell):
    def __init__(self):
        super(MLP, self).__init__()
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
```


### 2. 更新模型参数

```python
import mindspore
from mindspore import nn, Tensor
import numpy as np

model = MLP()
optimizer = nn.SGD(model.trainable_params(), 1e-3)
grad_fn = mindspore.value_and_grad(model, None, optimizer.parameters, has_aux=False)

def train():
    total_step = 10
    x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))

    for current_step in range(total_step):
        loss, grads = grad_fn(x, y)
        optimizer(grads)
        print(f"loss: {loss:>7f}  [{current_step + 1:>5d}/{total_step:>5d}]")
```

### 2.1. 使用 JIT(Just-In-Time) 编译加速

```python
import mindspore
from mindspore import nn, Tensor
import numpy as np

model = MLP()
optimizer = nn.SGD(model.trainable_params(), 1e-3)
grad_fn = mindspore.value_and_grad(model, None, optimizer.parameters, has_aux=False)

@mindspore.jit
def train_step(x, y):
    loss, grads = grad_fn(x, y)
    optimizer(grads)
    return loss

def train():
    total_step = 10
    x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))

    for current_step in range(total_step):
        loss = train_step(x, y)
        print(f"loss: {loss:>7f}  [{current_step + 1:>5d}/{total_step:>5d}]")
```

### 2.2. 如果训练流程比较复杂也可以将 `train_step` 包装为一个 `cell` 进行使用

#### 简单场景

```python
from mindspore import nn, ops, Tensor
import numpy as np


class MyTrainStep(nn.Cell):
    def __init__(self, network: nn.Cell, optimizer: nn.Optimizer):
        super().__init__(auto_prefix=False)

        self.network = network
        self.optimizer = optimizer
        
        self.network.set_train()
        self.network.set_grad()

        self.grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(self.network, optimizer.parameters)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, 1.0)
        grads = self.grad_fn(*inputs, sens)
        
        self.optimizer(grads)

        return loss


model = MLP()
optimizer = nn.SGD(model.trainable_params(), 1e-3)
train_step = MyTrainStep(model, optimizer)


def train():
    total_step = 10
    x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))

    for current_step in range(total_step):
        loss = train_step(x, y)
        print(f"loss: {loss:>7f}  [{current_step + 1:>5d}/{total_step:>5d}]")
```


#### 在上面的基础上增加 数据并行(data parallel)/梯度缩放(grad scale)/溢出检测 功能


```python
import mindspore
from mindspore import nn, ops, Tensor
import numpy as np

class MyTrainStep(nn.Cell):
    def __init__(
        self,
        network: nn.Cell,
        optimizer: nn.Optimizer,
        loss_scale: float = 1.0,
    ):
        super().__init__(auto_prefix=False)

        self.network = network
        self.optimizer = optimizer

        self.network.set_train()
        self.network.set_grad()

        self.grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(self.network, optimizer.parameters)

        # scaler and reduce
        _is_parallel = mindspore.context.get_auto_parallel_context("parallel_mode") == mindspore.ParallelMode.DATA_PARALLEL
        if _is_parallel:
            mean = mindspore.context.get_auto_parallel_context("gradients_mean")
            degree = mindspore.context.get_auto_parallel_context("device_num")
            self.reducer = nn.DistributedGradReducer(network.trainable_params(), mean, degree)
        else:
            self.reducer = nn.Identity()
        from mindspore.amp import StaticLossScaler, all_finite
        self.scaler = StaticLossScaler(scale_value=loss_scale)
        self.all_finite = all_finite
        self.all_finite_reducer = ops.AllReduce() if _is_parallel else nn.Identity()

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.scaler.scale_value)
        grads = self.grad_fn(*inputs, sens)
        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)

        finite = self.all_finite(unscaled_grads)
        finite = ops.equal(
            self.all_finite_reducer(finite.to(mindspore.int32)), self.all_finite_reducer(ops.ones((), mindspore.int32))
        ).to(mindspore.bool_)
        self.scaler.adjust(finite)

        if finite:
            loss = self.optimizer(unscaled_grads)

        is_overflow = not finite

        return loss, is_overflow


model = MLP()
optimizer = nn.SGD(model.trainable_params(), 1e-3)
train_step = MyTrainStep(model, optimizer, loss_scale=1.0)


def train():
    total_step = 10
    x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))

    for current_step in range(total_step):
        loss, is_overflow = train_step(x, y)
        print(f"loss: {loss:>7f}, is_overflow: {is_overflow} [{current_step + 1:>5d}/{total_step:>5d}]")
```
