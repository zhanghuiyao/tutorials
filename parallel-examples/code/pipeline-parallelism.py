import time
import numpy as np

import mindspore
from mindspore import context, nn, ops, Tensor, Parameter
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.communication.management import init
from mindspore.nn.utils import no_init_parameters
    

context.set_context(mode=context.GRAPH_MODE)
mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=4)
init()


class Mlp(nn.Cell):
    def __init__(self, num_layers: int = 8, in_channel: int = 512, out_channel: int = 512, num_labels: int = 1):
        super().__init__()
        
        # 1. & 2.
        # layers = [nn.Dense(in_channel, out_channel, activation="relu", has_bias=False)]
        # for _ in range(num_layers-1):
        #     layers.append(
        #         nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        #     )
        # self.layers = nn.CellList(layers)

        # 3.
        self.layer0 = nn.Dense(in_channel, out_channel, activation="relu", has_bias=False)
        self.layer1 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer2 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer3 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer4 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer5 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer6 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)
        self.layer7 = nn.Dense(out_channel, out_channel, activation="relu", has_bias=False)



        self.loss_fn = nn.MSELoss()

    def construct(self, x: Tensor, labels: Tensor = None):
        
        """
        x       : (bs, seq, channel)
        labels  : (bs, seq, channel)
        """

        # 1.
        # for layer in self.layers:
        #     x = layer(x)
        
        # 2.
        # x = self.layers[0](x)
        # x = self.layers[1](x)
        # x = self.layers[2](x)
        # x = self.layers[3](x)
        # x = self.layers[4](x)
        # x = self.layers[5](x)
        # x = self.layers[6](x)
        # x = self.layers[7](x)

        # 3.
        x = self.layers0(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.layers5(x)
        x = self.layers6(x)
        x = self.layers7(x)


        loss = self.loss_fn(x, labels)

        return loss


net = Mlp(num_layers=8)
optimizer = nn.AdamWeightDecay(net.trainable_params())


# pipeline-parallelism setting
# stage_config = {
#     "layers.0": 0, "layers.1": 0,   # stage 0
#     "layers.2": 1, "layers.3": 1,   # stage 1
#     "layers.4": 2, "layers.5": 2,   # stage 2
#     "layers.6": 3, "layers.7": 3, "loss_fn": 3  # stage 3
# }
stage_config = {
    "layers0": 0, "layers1": 0,   # stage 0
    "layers2": 1, "layers3": 1,   # stage 1
    "layers4": 2, "layers5": 2,   # stage 2
    "layers6": 3, "layers7": 3, "loss_fn": 3  # stage 3
}
pp_net = nn.PipelineCell(net, micro_size=4, stage_config=stage_config)
pp_net = AutoParallel(pp_net, parallel_mode="semi_auto")
pp_net.full_batch = True
pp_net.pipeline(stages=4, scheduler="1f1b")


grad_fn = ops.value_and_grad(pp_net, None, optimizer.parameters)
pp_grad_reducer = nn.PipelineGradReducer(optimizer.parameters)


@mindspore.jit
def train_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads


for i in range(10):

    x, y = Tensor(np.random.randn(4, 512, 512), mindspore.float32), Tensor(np.ones((4, 512, 512)), mindspore.float32)

    s_time = time.time()
    
    loss, grads = train_step(x, y)
    
    print(f"step: {i}, loss: {loss}, time cost: {(time.time()-s_time)*1000:.2f} ms")
