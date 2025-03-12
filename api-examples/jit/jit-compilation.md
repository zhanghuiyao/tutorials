
# 即时编译 (Just-in-time Compilation)

在本节中，我们将进一步探讨MindSpore的工作原理，以及如何使其高效运行。我们将讨论 `mindspore.jit()` 转换，它将执行对MindSpore Python函数的即时编译（just-in-time compilation），以便在后续过程中高效执行。这个过程会花费一些时间，它发生在函数第一次执行的时候。


## 版本

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   master-0303-a526321d    |    24.1.RC2    | 7.3.0.1.231 |   8.0.0.beta1    |


## 目录

- [对函数进行JIT编译]()
- [更多的用法]()
- [我们做的一些实验]()


## 1. 对函数进行JIT编译

### 1.1. 函数定义

```python
from mindspore import Tensor

def f(a: Tensor, b: Tensor, c: Tensor):
    return a * b + c
```

### 1.2. 使用 `mindspore.jit` 包装

```python
import mindspore

jitted_defult_f = mindspore.jit(f)
```

### 1.3. 构造数据并运行

```python
import time

f_input = [Tensor(np.full((2, 3), i), mindspore.float32) for i in range(3)]
print(f"function input: {f_input}")

# 运行原始函数1000次
s_time = time.time()
for _ in range(1000):
    out = f(*f_input)
print(f"{out=}, time cost: {(time.time()-s_time)*1000:.2f}ms")


# 运行jit后的函数1000次
s_time = time.time()
for _ in range(1000):
    out = jitted_defult_f(*f_input)
print(f"{out=}, time cost: {(time.time()-s_time)*1000:.2f}ms")
```


## 2. 更高级的用法

### 2.1. 常用配置介绍

> `mindspore.jit`接口详情见[文档描述](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html)

- capture_mode: 用于指定创建`图`的方式（如：`ast`通过解析python构建， `bytecode`通过解析Python字节码构建， `trace`通过追踪Python代码的执行进行构建。）
- jit_level: 用于控制编译优化的级别。（如： 默认O0， 使用更多的优化O1）
- fullgraph: 是否将整个函数编译为`图`，默认为False，jit会尽可能兼容函数中的python语法，打开为True一般可以获得更好的性能，但对语法要求更高。
- backend: 用于指定编译的后端。


### 2.2. 使用方法

```python
# 使用ast的方式构建图
jitted_by_ast_and_levelO0_f = mindspore.jit(f, capture_mode="ast", jit_level="O0") # 这个是默认配置，跟上面的jitted_defult_f是一样的
jitted_by_ast_and_levelO1_f = mindspore.jit(f, capture_mode="ast", jit_level="O1")
jitted_by_ast_and_ge_f = mindspore.jit(f, capture_mode="ast", backend="GE")

# 使用的bytecode方式构建图
jitted_by_bytecode_and_levelO0_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O0")
jitted_by_bytecode_and_levelO1_f = mindspore.jit(f, capture_mode="bytecode", jit_level="O1")
jitted_by_bytecode_and_ge_f = mindspore.jit(f, capture_mode="bytecode", backend="GE")


# 使用的trace方式构建图，trace不支持直接通过mindspore.jit(f, capture_mode="trace", ...)的方式转换
@mindspore.jit(capture_mode="trace", jit_level="O0")
def jitted_by_trace_and_levelO0_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", jit_level="O1")
def jitted_by_trace_and_levelO1_f(a, b, c):
    return a * b + c

@mindspore.jit(capture_mode="trace", backend="GE")
def jitted_by_trace_and_ge_f(a, b, c):
    return a * b + c

# 尝试使用full_graph (这里以ast为例子)
jitted_by_ast_and_levelO0_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O0", fullgraph=True)
jitted_by_ast_and_levelO1_fullgraph_f = mindspore.jit(f, capture_mode="ast", jit_level="O1", fullgraph=True)
jitted_by_ast_and_ge_fullgraph_f = mindspore.jit(f, capture_mode="ast", backend="GE", fullgraph=True)


# 用字典记录，方便后续调用
function_dict = {
    "function ": f,
    
    "function jitted by ast and levelO0": jitted_defult_f,
    "function jitted by ast and levelO1": jitted_by_ast_and_levelO1_f,
    "function jitted by ast and ge": jitted_by_ast_and_ge_f,
    
    "function jitted by bytecode and levelO0": jitted_by_bytecode_and_levelO0_f,
    "function jitted by bytecode and levelO1": jitted_by_bytecode_and_levelO1_f,
    "function jitted by bytecode and ge": jitted_by_bytecode_and_ge_f,

    "function jitted by trace and levelO0": jitted_by_trace_and_levelO0_f,
    "function jitted by trace and levelO1": jitted_by_trace_and_levelO1_f,
    "function jitted by trace and ge": jitted_by_trace_and_ge_f,

    "function jitted by ast and levelO0 fullgraph": jitted_by_ast_and_levelO0_fullgraph_f,
    "function jitted by ast and levelO1 fullgraph": jitted_by_ast_and_levelO1_fullgraph_f,
    "function jitted by ast and ge fullgraph": jitted_by_ast_and_ge_fullgraph_f
}
```

> ⚠️ 注意：当构建图的方式选择为trace的时候不支持直接通过mindspore.jit(f, capture_mode="trace", ...)的方式转换，需要通过装饰器用法对函数进行包装。


### 2.3. 运行

```python
for s, f in function_dict.items():
    s_time = time.time()
    
    out = f(*f_input)

    time_to_prepare = time.time() - s_time
    s_time = time.time()

    # 每个函数都运行1000次
    for _ in range(1000):
        out = f(*f_input)
    
    time_to_run_thousand_times = time.time() - s_time

    print(f"{s}, out shape: {out.shape}, time to prepare: {time_to_prepare:.2f}s, time to run thousand times: {time_to_run_thousand_times:.2f}s")
```

## 3. 我们做的一些实验

⚠️ 注意：不同的软硬件条件下，可能会有很大的差异，以下结果仅供参考。

### 3.1. 测试一个简单的函数

我们定义一个函数 `funtion(a,b,c)=a*b+c`, 并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行脚本，

```shell
export GLOG_v=3  # 可选，设置更高的MindSpore日志以减少一些打印让结果看起来更美观
python -u simple_funtion.py
```

结果如下：

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


#### ⚠️ 注意:

1. 上述结果因设备和设备状态的不同而差异很大，数据仅供参考。
2. *准备时间(time to prepare)：潜在的jitted对象重用和设备内存拷贝可能会导致比较不准确。
3. *运行一千次的时间(time to run thousand times)：潜在的异步执行操作可能会导致测试时间不准确。

#### ⚠️ 一些限制:

1. mindspore.jit不能在终端中使用临时源代码进行编译，必须作为`.py`文件运行。
2. 当构建图的方式选择为trace的时候不支持直接通过`mindspore.jit(f, capture_mode="trace", ...)`的方式转换，需要通过装饰器`@mindspore.jit(capture_mode="trace", ...)`用法对函数进行包装。


### 3.2. 测试一个简单的卷积模块 (Conv Module)

我们定义一个在经典网络`resnet`中使用到的核心模块`BasicBlock`, 并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行脚本，

```shell
python -u simple_conv.py
```

结果如下：

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~6.53s | ~1.68s    |
| true  | O0    | ast        | ms_backend    | false | ~0.84s | **~0.88s**    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~2.02s | ~4.22s    |
| true  | O0    | ast        | ms_backend    | false | ~0.82s | **~1.89s**    |

#### ⚠️ 注意:

1. 上述结果因设备和设备状态的不同而差异很大，数据仅供参考。
2. *准备时间(time to prepare)：潜在的jitted对象重用和设备内存拷贝可能会导致比较不准确。
3. *运行一千次的时间(time to run thousand times)：潜在的异步执行操作可能会导致测试时间不准确。


### 3.3. 测试一个简单的注意力模块 (Attention Module)


我们定义一个在经典网络`llama3`中使用到的核心模块`LlamaAttention`, 并使用 `mindspore.jit` 进行转换, 可以通过以下命令运行脚本，

```shell
python -u simple_attention.py
```

结果如下：

**forward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~5.76s | ~4.87s    |
| true  | O0    | ast        | ms_backend    | false | ~1.65s | **~3.41s**    |

**forward + backward**

| enable jit | jit level | capture mode | backend | fullgraph | *time to prepare | *time to run thousand times |
| --- | --- | --- | --- | --- | --- | --- |
| false | -     | -          | -             | -     | ~0.15s | ~10.12s    |
| true  | O0    | ast        | ms_backend    | false | ~1.68s | **~6.38s**    |

#### ⚠️ 注意:

1. 上述结果因设备和设备状态的不同而差异很大，数据仅供参考。
2. *准备时间(time to prepare)：潜在的jitted对象重用和设备内存拷贝可能会导致比较不准确。
3. *运行一千次的时间(time to run thousand times)：潜在的异步执行操作可能会导致测试时间不准确。

