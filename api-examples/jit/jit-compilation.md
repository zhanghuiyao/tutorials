
# 即时编译 (Just-in-time Compilation)

在本节中，我们将进一步探讨MindSpore的工作原理，以及如何使其高效运行。我们将讨论 `mindspore.jit()` 转换，它将执行对MindSpore Python函数的即时编译（just-in-time compilation），以便在后续过程中高效执行。

这个过程会花费一些时间，它发生在函数第一次执行的时候。


## 版本

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   master-0303-a526321d    |    24.1.RC2    | 7.3.0.1.231 |   8.0.0.beta1    |


## 目录

- [对函数进行JIT编译]()
- [JIT的一些限制]()
- [更高级的用法]()
- [我们做的一些实验]()


## 1. 对函数进行JIT编译


## 2. JIT的一些限制

⚠️ 注意：下面这些是我们在常用场景下发现的，如果你还发现了别的限制，请告诉我们。

//in
- dynamic shape
- dict/tuple 输入
- 函数/Cell/类及对象 作为输入

//out
- 返回None,Dict,自定义类对象作为输出

//compute
- 控制流
- 第三方库调用 (e.g. math, sklearn, ...)

## 3. 更高级的用法


## 4. 我们做的一些实验

⚠️ 注意：不同的软硬件条件下，可能会有很大的差异，以下结果仅供参考。






## 1. test simple function examples

we define `funtion(a,b,c)=a*b+c`, and warp it by `mindspore.jit`, run by `GLOG_v=3 python -u simple_funtion.py`, result is follow:

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~4.16s | **~0.06s**    |
||
| true  | O0    | ast        | ms_backend    | false | ~0.21s | **~0.50s**   |
| true  | O1    | ast        | ms_backend    | false | ~0.03s | ~0.51s   |
| true  | -     | ast        | ge            | false | ~1.01s | ~1.04s   |
||
| true  | O0    | bytecode   | ms_backend    | false | ~0.13s | **~0.72s**   |
| true  | O1    | bytecode   | ms_backend    | false | ~0.00s | ~0.74s   |
| true  | -     | bytecode   | ge            | false | ~0.00s | ~0.74s   |
||
| true  | O0    | trace      | ms_backend    | false | ~0.17s | ~3.34s   |
| true  | O1    | trace      | ms_backend    | false | ~0.15s | **~2.38s**   |
| true  | -     | trace      | ge            | false | ~0.17s | ~3.49s   |
||
| true  | O0    | ast        | ms_backend    | true  | ~0.02s | **~0.56s**   |
| true  | O1    | ast        | ms_backend    | true  | ~0.03s | **~0.56s**   |
| true  | -     | ast        | ge            | true  | ~0.14s | ~1.03s   |


#### ⚠️ Note:

0. the above results vary greatly for different devices and device states, data only for reference.
1. *time to prepare, potential jitted object reuse and device memory copy may lead to inaccurate comparison.
2. *time to run thousand times, potential asynchronous execution operations may lead to inaccurate testing times.


#### ⚠️ Limitations:

1. mindspore.jit can not compile with temporary source code in terminal, must run as a `.py` file.
2. jit by trace must be use the registry method?
3. jit by trace do not support python `*input`?


## 2. test simple module examples

### 2.1. simple `conv` module

we define `BasicBlock` which is use in `resnet`, and warp it by `mindspore.jit`, run by `GLOG_v=3 python -u simple_conv.py`, result is follow:

#### 2.1.1. forward

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~6.25s | ~1.73s    |
| true  | O0    | ast        | ms_backend    | false | ~0.34s | **~0.44s**    |

#### 2.1.2. forward + backward

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~1.49s | ~5.19s    |
| true  | O0    | ast        | ms_backend    | false | ~0.62s | **~0.52s**    |

#### ⚠️ Note:

0. the above results vary greatly for different devices and device states, data only for reference.
1. *time to prepare, potential jitted object reuse and device memory copy may lead to inaccurate comparison.
2. *time to run thousand times, potential asynchronous execution operations may lead to inaccurate testing times.


### 2.2. simple `attention` module

we define `LlamaBlock` which is use in `llama`, and warp it by `mindspore.jit`, run by `GLOG_v=3 python -u simple_xx.py`, result is follow:


