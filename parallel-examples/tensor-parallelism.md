# MindSpore张量并行实践指南

本教程基于 MindSpore 2.5.0 版本，演示如何使用张量并行技术加速模型训练。示例代码在 4张 Ascend 910* 卡运行。


## 0. 张量并行介绍 (introduction)

张量并行技术是分布式大模型训练的核心方法之一，通过对单个张量运算（如矩阵乘法）进行拆分，实现计算与存储的细粒度并行。其核心思想是将大型权重矩阵按行或列分割至不同设备，各设备独立处理局部计算后，通过集合通信（如All-Reduce）同步结果。例如，在Transformer模型中，多头注意力层的参数矩阵可水平拆分到多个NPU/GPU，每个GPU计算部分注意力头，最终拼接输出。该技术显著降低单设备内存压力，支持大模型训练。


## 1. 环境配置 (init and setting)

```python
# 设置运行模式为图模式
context.set_context(mode=context.GRAPH_MODE)

# 配置半自动并行策略
mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.SEMI_AUTO_PARALLEL)

# 初始化分布式环境
init()
```

## 2. 模型定义与并行配置 (define)


### 2.1. 模型定义

```python
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

# 创建模型 (create model)
net = TwoMatmul()
```

### 2.2. Primitive算子切分策略配置

```python
# 将matmul1的第一个输入Tensor切分为(2,2)的分布，将第二个输入Weight切分为(2,1)的分布
net.matmul1.shard(((2, 2), (2, 1)))   
net.relu1.shard(((2, 1),))
net.matmul2.shard(((1, 4), (4, 1)))
net.relu2.shard(((4, 1),))
```

**说明：**
以 `net.matmul1.shard(((2,2), (2,1)))` 为例，`net.matmul1`的两个输入分别为`x`和`weight1`，原始的形状分别为`(4, 512)`和`(512, 512)`，切分后的形状分别为`(2, 256)`和`(256, 512)`


## 3. 训练 (training)

```python
# 初始化优化器和梯度函数
optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=0.01)
grad_fn = ops.value_and_grad(net, None, optimizer.parameters)
grad_reducer = nn.Identity()

@mindspore.jit
def train_step(inputs, target):
    loss, grads = grad_fn(inputs, target)
    grads = grad_reducer(grads)
    optimizer(grads)
    return loss, grads

# 构造示例数据
x = Tensor(np.random.randn(4, 512), mindspore.float32)
y = Tensor(np.ones((4, 512)), mindspore.float32)

# 训练循环
for i in range(100):
    loss, grads = train_step(x, y)
    if (i+1) % 10 == 0:
        print(f"step: {i+1}, loss: {loss}")

# 打印当前卡模型的权重、优化器状态和梯度张量的大小
print(f"{net.weight1.shape=}, {optimizer.moments1[0].shape=}, {optimizer.moments2[0].shape=}, {grads[0].shape=}")   # matmul1 weight shard to (2, 1)
print(f"{net.weight2.shape=}, {optimizer.moments1[1].shape=}, {optimizer.moments2[1].shape=}, {grads[1].shape=}")   # matmul2 weight shard to (4, 1)
```


## 4. 运行示例代码 (running)

```shell
# 运行示例代码，使用前4张卡
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 msrun --bind_core=True --worker_num=4 --local_worker_num=4 --master_port 9001 --log_dir=outputs/parallel_logs \
python -u code/pipeline-parallelism.py

# 查看日志
tail -f outputs/parallel_logs/worker_0.log
```

输出打印：
```
step: 10, loss: 0.9715, time cost: 5.63 ms
step: 20, loss: 0.9698, time cost: 5.51 ms
step: 30, loss: 0.9681, time cost: 5.58 ms
...

net.weight1.shape=(256, 512), grads[0].shape=(256, 512)
net.weight2.shape=(128, 512), grads[1].shape=(128, 512)
```


## 5. 了解更多

可以在[MindSpore网站](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/operator_parallel.html)搜索“张量并行”。