# `TrainStep` in MindONE.diffusers

MindSpore 框架使用[函数式自动微分](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html)的设计理念，提供了更接近于数学语义的自动微分接口`grad`和`value_and_grad`。我们需要根据计算图描述的计算过程构造计算函数，实现一个 `loss = function(data)` 前向流程的函数封装，然后使用 `value_and_grad` 接口获得自动微分函数，用于计算梯度。

通常我们先基于模型定义一个 `ModelWithLoss` 类(是一个`mindspore.nn.Cell`类)，输入数据和计算loss。随后构造 `TrainOneStepWrapper` 类（也是一个 `mindspore.nn.Cell` 类）作为单步训练流程的封装，loss的前向计算、自动反向微分、优化器更新等整个流程丢进该类的`construct`方法中。随后在数据迭代循环中直接调用 `TrainOneStepWrapper` 类的实例即可。

`MindONE.diffusers` 组件提供了 `TrainStep` 类，基于前向流程进行前反向和优化器更新的更高阶封装，进一步地简化了训练流程的定义。直接在训练脚本中继承`TrainStep` ，定义`TrainStepForMyNet` 类，并定义其前向计算loss的过程到 `forward` 方法。


以下是使用 `MindONE.diffusers` 的 `TrainStep` 类的基本用法，只需要实现自定义网络的前向方法即可。

```python
from mindspore import ops, nn
from mindspore.amp import DynamicLossScaler
from mindone.diffusers.training_utils import TrainStep

# 继承 `TrainStep`, 定义自定义网络的前向计算 loss 的过程
# `TrainStep` 的 `construct` 方法已封装实现好前向、反向、更新的全流程
class TrainStepForMyNet(TrainStep):
    def forward(self, x):
        y = self.model(x)
        loss = ops.sum(y)
        loss = self.scale_loss(loss)
        return loss, y

# 定义网络，优化器
model = nn.Dense(10, 10)
optim = nn.AdamWeightDecay(model.trainable_params())

# 实例化 `TrainStepForMyNet`
train_step = TrainStepForMyNet(
    model,
    optim,
    loss_scaler=DynamicLossScaler(2.0**16, 2, 2000),
    max_grad_norm=1.0,
    gradient_accumulation_steps=2,
    gradient_accumulation_kwargs={"length_of_dataloader": 3}
)

# 在迭代训练循环中调用 `TrainStepForMyNet` 的实例
for epoch in range(2):
    for batch in range(3):
        inputs = ops.randn(8, 10)
        outputs = train_step(inputs)
```

需要特别注意的是，抽象类 `TrainStep` 的 `construct` 里写好了 unscales the loss，即，`loss = self.unscale_loss(outputs[0])`。因此我们在自定义的实现类 `TrainStepForMyNet` 的 `forward` 实现里，计算好 loss 后记得 scale the loss， 即 调用 `loss = self.scale_loss(loss)`， 否则会产生错误计算。


 ```python
class TrainStep(...)

     def scale_loss(self, loss):
        loss = loss / self.grad_accumulator.gradient_accumulation_steps
        loss = self.grad_scaler.scale(loss)
        return loss

    def unscale_loss(self, loss):
        return self.grad_scaler.unscale(loss)

    def construct(self, *inputs):
        outputs, grads = self.forward_and_backward(*inputs)
        grads = self.grad_accumulator.step(grads)
        ...
        # The first item of outputs is loss. Unscales the loss for outside logging.
        loss = self.unscale_loss(outputs[0])
        outputs = (loss,) + outputs[1:]
        return outputs
```


扩展阅读
- [Flux dreambooth lora 单步训练实现](flux_dreambooth_lora_train_step.md)
- Flux controlnet 单步训练实现（todo）