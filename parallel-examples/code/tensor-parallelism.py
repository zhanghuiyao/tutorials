import time
import numpy as np

import mindspore
from mindspore import context, nn, ops, Tensor, Parameter
from mindspore.communication.management import init
from mindspore.common.initializer import initializer


context.set_context(mode=context.GRAPH_MODE)
mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL)
init()


class TwoMatmul(nn.Cell):
    def __init__(self):
        super().__init__()

        self.weight1 = Parameter(initializer("normal", [512, 512], mindspore.float32))
        self.weight2 = Parameter(initializer("normal", [512, 512], mindspore.float32))
        
        self.matmul1 = ops.MatMul()
        self.relu1 = ops.ReLU()
        self.matmul2 = ops.MatMul()
        self.relu2 = ops.ReLU()

        self.loss_fn = nn.MSELoss()

    def construct(self, x, labels):
        x = self.matmul1(x, self.weight1)
        x = self.relu1(x)
        x = self.matmul2(x, self.weight2)
        x = self.relu2(x)

        loss = self.loss_fn(x, labels)

        return loss


net = TwoMatmul()

# tensor-parallelism primitive shard setting, 
net.matmul1.shard(((2, 2), (2, 1)))
net.relu1.shard(((2, 1),))
net.matmul2.shard(((1, 4), (4, 1)))
net.relu2.shard(((4, 1),))


net.set_train()
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
grad_fn = ops.value_and_grad(net, None, optimizer.parameters)
grad_reducer = nn.Identity()


@mindspore.jit
def train_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = grad_reducer(grads)
    optimizer(grads)
    return loss, grads


x, y = Tensor(np.random.randn(4, 512), mindspore.float32), Tensor(np.ones((4, 512)), mindspore.float32)

for i in range(100):
    
    s_time = time.time()

    loss, grads = train_step(x, y)
    
    if (i+1) % 10 == 0:
        print(f"step: {i+1}, loss: {loss}, time cost: {(time.time()-s_time)*1000:.2f} ms")


print(f"{net.weight1.shape=}, {grads[0].shape=}")   # matmul1 weight shard to (2, 1)
print(f"{net.weight2.shape=}, {grads[1].shape=}")   # matmul2 weight shard to (4, 1)
