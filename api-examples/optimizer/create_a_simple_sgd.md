# 从零开始写一个简单的sgd优化器

> 备注：以下示例使用的是 MindSpore 2.4.0 版本

## 目录

- [从零开始创建一个sgd优化器](#section1)
- [加速](#section2)


<a id="section1"></a>
## 从零开始创建一个`sgd`优化器

- 第一步：定义`sgd`优化过程

    ```python
    from mindspore import ops
            
    def sgd_update(weights, grads, lr):
        for w, dw in weights, grads:
            ops.assign(w, w - lr * dw)
    ```

- 第二步：更新模型参数

    > `Net` 模型定义请[参考](../../model-examples/create_a_simple_nerual_network.md)
    
    ```python
    import mindspore
    from mindspore import Tensor
    import numpy as np
    
    net = Net()
    net.trainable_params()
    grad_fn = mindspore.value_and_grad(net, None, net.trainable_params(), has_aux=False)
    
    def train():
        total_step = 10
        x, y = Tensor(np.random.randn(1, 1, 16, 16)), Tensor(np.ones(1))
    
        for current_step in range(total_step):
            loss, grads = grad_fn(x, y)
            sgd_update(net.trainable_params(), grads, 0.01)
            print(f"loss: {loss:>7f}  [{current_step + 1:>5d}/{total_step:>5d}]")
    ```

- 当然我们也可以将 `sgd` 优化器定义为一个类对象

    ```python
    from mindspore import nn, ops
    
    class SGD(nn.Cell):
        def __init__(self, weights, lr):
            super(SGD, self).__init__()
            self.weights = weights
            self.lr = lr
    
        def construct(self, grads):
            for w, dw in self.weights, grads:
                ops.assign(w, w - self.lr * dw)
    ```

<a id="section2"></a>
## 加速

- 使用 `JIT(Just-In-Time)` 编译加速

  ```python
  import mindspore
  from mindspore import nn, ops
  
  class SGD(nn.Cell):
      def __init__(self, weights, lr):
          super(SGD, self).__init__()
          self.weights = weights
          self.lr = lr
  
      @mindspore.jit
      def construct(self, grads):
          for w, dw in self.weights, grads:
              ops.assign(w, w - self.lr * dw)
  ```

- 使用 `mindspore.ops.HyperMap` 操作替换 `for` 循环

  ```python
  import mindspore
  from mindspore import nn, ops
  
  sgd_update = ops.MultitypeFuncGraph("_sgd_update")
  
  @sgd_update.register("Tensor", "Tensor", "Tensor")
  def run_sgd_update(lr, grad, weight):
      """Apply sgd optimizer to the weight parameter using Tensor."""
      success = True
      ops.depend(success, ops.assign(weight, weight - lr * grad))
      return success
  
  class SGD(nn.Cell):
      def __init__(self, weights, lr):
          super(SGD, self).__init__()
          self.weights = weights
          self.lr = lr
          self.hyper_map = ops.HyperMap()
  
      @mindspore.jit
      def construct(self, grads):
          return self.hyper_map(
              ops.partial(sgd_update, self.lr),
              self.weights,
              grads
          )
  ```
