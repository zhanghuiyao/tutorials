# 使用mindspore对函数求导

> 备注：以下示例使用的是 MindSpore 2.4.0 版本

## 目录

- [使用`mindspore.grad`对函数进行求导](#section1)
- [使用`mindspore.grad`计算线性逻辑回归的梯度](#section2)
- [使用`mindspore.value_and_grad`计算梯度与损失](#section3)

<br>

<a id="section1"></a>
## 使用`mindspore.grad`对函数进行求导

`f(x) = x^3 - 2x + 1` 的导数可以计算为：

$$
\begin{aligned}
& f(x) = x^3 - 2x + 1 \\
& f'(x) = 3x - 2 \\
& f''(x) = 3
\end{aligned}
$$

使用MindSpore可以简单的表示为：

```python
from mindspore import grad, Tensor

f = lambda x: x**3 - 2*x + 1

dfdx = grad(f)
d2fdx = grad(grad(f))
```

当`x=1`时，对上述内容进行验证可以得到：

$$
\begin{aligned}
& f(1) = 0 \\
& f'(1) = 1 \\
& f''(1) = 3
\end{aligned}
$$

在MindSpore中运行:

```python
print(f(Tensor(1.0)))
print(dfdx(Tensor(1.0)))
print(d2fdx(Tensor(1.0)))
```

<a id="section2"></a>
## 使用`mindspore.grad`计算线性逻辑回归的梯度

首先，我们做如下定义：

```python
from mindspore import ops, grad, Tensor, set_seed

set_seed(0)

inputs = ops.randn((4, 3))
targets = Tensor([True, True, False, True])

W, b = ops.randn((3,)), ops.randn((1,))

def loss(inputs, targets, W, b):
    preds = ops.sigmoid(ops.matmul(inputs, W) + b)
    logit_loss = preds * targets + (1 - preds) * (1 - targets)
    return -ops.log(logit_loss).sum()

print(f"inputs: {inputs}")
print(f"targets: {targets}")
print(f"W: {W}")
print(f"b: {b}")
```

分别计算`W`和`b`等输入的梯度：

```python
x_grad = grad(loss, grad_position=0)(inputs, targets, W, b)
print(f'x_grad: {x_grad}')

y_grad = grad(loss, 1)(inputs, targets, W, b)
print(f'y_grad: {y_grad}')

W_grad = grad(loss, 2)(inputs, targets, W, b)
print(f'W_grad: {W_grad}')

b_grad = grad(loss, 3)(W, b)
print(f'b_grad: {b_grad}')
```

当然也可以一次性计算所需要的梯度：

```python
(W_grad, b_grad) = grad(loss, (2, 3))(inputs, targets, W, b)
print(f'W_grad: {W_grad}')
print(f'b_grad: {b_grad}')
```

如果函数输出多个`loss`，计算的梯度是所有`loss`对输入的导数，

```python
def multi_loss(inputs, targets, W, b):
    loss1 = loss(inputs, targets, W, b)
    loss2 = (W ** 2).sum()
    return loss1, loss2

(W_grad, b_grad) = grad(multi_loss, (2, 3))(inputs, targets, W, b)
print(f'W_grad: {W_grad}, b_grad: {b_grad}')
```

如果只想计算`loss1`的梯度，可以尝试用`ops.stop_gradient`进行截断：

```python
def multi_loss_2(inputs, targets, W, b):
    loss1 = loss(inputs, targets, W, b)
    loss2 = (W ** 2).sum()
    return loss1, ops.stop_gradient(loss2)

(W_grad, b_grad) = grad(multi_loss_2, (2, 3))(inputs, targets, W, b)
print(f'W_grad: {W_grad}, b_grad: {b_grad}')
```

或者，也可以通过设置`has_aux=True`排除除了第一个以外的输出对梯度的影响：

```python
(W_grad, b_grad) = grad(multi_loss, (2, 3), has_aux=True)(inputs, targets, W, b)
print(f'W_grad: {W_grad}, b_grad: {b_grad}')
```

`mindspore.grad`接口的更多细节可以参考 [MindSpore Docs](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.grad.html)


<a id="section3"></a>
## 使用`mindspore.value_and_grad`计算梯度与损失

计算线性逻辑回归函数的梯度，并获取`loss`：

```python
from mindspore import value_and_grad

loss, (W_grad, b_grad) = value_and_grad(loss, (2, 3))(inputs, targets, W, b)

print(f"loss: {loss}")
print(f'W_grad: {W_grad}, b_grad: {b_grad}')
```

`mindspore.value_and_grad`接口的更多细节可以参考 [MindSpore Docs](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.value_and_grad.html)
