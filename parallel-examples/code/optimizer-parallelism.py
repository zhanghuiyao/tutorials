import time
import numpy as np

import mindspore
from mindspore import context, nn, ops, Tensor
from mindspore.communication.management import init
    

context.set_context(mode=context.GRAPH_MODE)
mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL, enable_parallel_optimizer=True)
init()


class Mlp(nn.Cell):
    
    # @mindspore.lazy_inline  # lazy_inline is not required in optimizer-parallelism
    def __init__(self, num_layers: int = 4, in_channel: int = 512, out_channel: int = 512):
        super().__init__()
        
        layers = [nn.Dense(in_channel, out_channel, activation="relu", has_bias=False)]
        for _ in range(num_layers-1):
            layers.append(
                nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
            )
        self.layers = nn.CellList(layers)

        self.loss_fn = nn.MSELoss()

    def construct(self, x: Tensor, labels: Tensor = None):

        for layer in self.layers:
            x = layer(x)

        loss = self.loss_fn(x, labels)

        return loss


net = Mlp(num_layers=4)

# (option) enable communication fusion
net.set_comm_fusion(1)

net.set_train()
optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=0.01)
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
        print(f"step: {i+1}, loss: {loss}, per step time: {(time.time()-s_time)*1000:.2f} ms")


print(f"{net.layers[0].weight.shape=}, {optimizer.moments1[0].shape=}, {optimizer.moments2[0].shape=}, {grads[0].shape=}")
print(f"{net.layers[1].weight.shape=}, {optimizer.moments1[1].shape=}, {optimizer.moments2[1].shape=}, {grads[1].shape=}")
print(f"{net.layers[2].weight.shape=}, {optimizer.moments1[2].shape=}, {optimizer.moments2[2].shape=}, {grads[2].shape=}")
print(f"{net.layers[3].weight.shape=}, {optimizer.moments1[3].shape=}, {optimizer.moments2[3].shape=}, {grads[3].shape=}")
