# MindSpore流水线并行实践指南

本教程基于MindSpore2.5版本，演示如何使用流水线并行技术加速模型训练。示例代码需要 4张 Ascend 910* 卡运行。


## 1. 流水线并行核心配置 (init and setting)

```python
# 设置运行模式为图模式
context.set_context(mode=context.GRAPH_MODE)

# 配置半自动并行策略
mindspore.set_auto_parallel_context(
    parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL,
    pipeline_stages=4,
    pipeline_config={
        'pipeline_scheduler': '1f1b',  # 使用1F1B调度策略
        'pipeline_interleave': True    # 启用流水线交错
    }
)

# 初始化分布式环境
init()
```


## 2. 模型定义与阶段划分 (define)

```python
class Mlp(nn.Cell):
    @mindspore.lazy_inline  # 必须使用lazy_inline装饰器
    def __init__(self, num_layers=4, in_channel=512, out_channel=512):
        super().__init__()
        layers = [nn.Dense(in_channel, out_channel, activation="relu", has_bias=False)]
        for _ in range(num_layers-1):
            layers.append(nn.Dense(out_channel, out_channel, activation="relu", has_bias=False))
        self.layers = nn.CellList(layers)
        self.loss_fn = nn.MSELoss()

    def construct(self, x, labels=None):
        for layer in self.layers:
            x = layer(x)
        return self.loss_fn(x, labels)

# 创建模型 (create model)
net = Mlp()

# 分配流水线阶段 (pipeline stage)
net.layers[0].pipeline_stage = 0  # 第1个Dense层分配到stage0
net.layers[1].pipeline_stage = 1  # 第2个Dense层分配到stage1 
net.layers[2].pipeline_stage = 2  # 第3个Dense层分配到stage2
net.layers[3].pipeline_stage = 3  # 第4个Dense层分配到stage3
net.loss_fn.pipeline_stage = 3    # 损失函数分配到stage3

# 封装为流水线模型
pp_net = nn.PipelineCell(net, micro_size=4)  # micro_batch_size=4
```

> ⚠️ 注意：用于进行流水线并行的模型需要使用 `@mindspore.lazy_inline` 装饰器。


## 3. 运行 (running)

```python
# 初始化优化器和梯度函数
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.001)
grad_fn = ops.value_and_grad(pp_net, None, optimizer.parameters)
pp_grad_reducer = nn.PipelineGradReducer(optimizer.parameters)

@mindspore.jit
def train_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = pp_grad_reducer(grads)
    optimizer(grads)
    return loss, grads

# 构造示例数据（batch-size必须能被micro-batch-size整除）
x = Tensor(np.random.randn(4, 512), mindspore.float32)
y = Tensor(np.ones((4, 512)), mindspore.float32)

# 训练循环
for i in range(100):
    loss, grads = train_step(x, y)
    if (i+1) % 10 == 0:
        print(f"step: {i+1}, loss: {loss}")
```


## 4. 运行结果示例

典型输出：
```
step: 10, loss: 0.8932, time cost: 152.34 ms
step: 20, loss: 0.7615, time cost: 148.91 ms
step: 30, loss: 0.6543, time cost: 149.12 ms
...
net.layers[get_rank()].weight.shape=(512, 512), grads[0].shape=(512, 512)
```


## 5. 关键注意事项

1. **设备分配**：需要保证`pipeline_stages`数量能被实际使用的GPU数量整除。
2. **数据规范**：batch_size必须能被micro_batch_size整除
3. **装饰器要求**：使用`@lazy_inline`装饰模型初始化方法
4. **调度策略**：1F1B调度策略适合大多数场景，也可尝试`gpipe`策略
5. **性能调优**：可通过调整`pipeline_interleave`和`micro_size`优化吞吐


## 6. 常见问题

**Q1: 如何验证流水线并行是否正确工作？**
A: 检查不同stage的GPU显存占用情况，各卡应有相似的显存使用量

**Q2: 出现"Shape mismatch"错误怎么办？**
A: 检查各stage的输入输出维度是否匹配，确保相邻stage的矩阵维度兼容

**Q3: 如何扩展更多流水线阶段？**
A: 增加pipeline_stages参数值，并相应调整模型层的stage分配

**Q4: 微批次大小如何选择？**
A: 通常设置为总批次大小的约数，可通过实验选择吞吐量最大的值
