## 精度设置优化 - 尝试更换可自动 upcast 计算的优化器

延续 [模型加载、LoRA层初始化与模块精度设置](flux_lora_load_models.md) 、 [模型加载、Controlnet层初始化与各模块精度设置](flux_controlnet_load_models.md) 对模块精度设置的讨论，以 `args.mixed_precision = bf16` 为例，我们知道以下信息：

- 不参与训练的模块，参数精度与运行精度均为半精度
- 训练模型参数精度设为全精度，运行精度通过 to_float 设为半精度

flux lora 训练：

| precision   | vae  | textencoders | transformers | LoRA layers |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | bf16         | bf16         | fp32        |
| computation | bf16 | bf16         | bf16         | bf16        |

flux controlnet 训练：

| precision   | vae  | textencoders | flux_transformer | flux_controlnet |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | bf16         | bf16         | fp32        |
| computation | bf16 | bf16         | bf16         | bf16        |


我们需要在训练脚本中保证其参数的精度是全精度的原因是，使用 mindspore 框架的优化器，比如 `nn.AdamWeightDecay`, 当前是按照参数的精度做梯度更新的，而不会在反向更新权重时自动 upcast。假如训练参数也设置成半精度，变成完全的半精度训练，在梯度更新时可能会导致溢出，无法正常训练。

但是 flux controlnet 微调比较耗性能，训练参数有 1.44B，在未讨论 ZeRO 等并行优化手段时，单卡用户可能会遇到 OOM 问题。我们可以尝试修改优化器，使得优化器在更新计算的时候可以自动 upcast 精度再更新，而存的权重可以保持半精度。虽然优化有限，但也可以节省出一点点显存。

我们可以暂时把 脚本中的 mindspore 框架的 `nn.AdamWeightDecay` 优化器更换成 mindcv 仓实现的adamw https://github.com/mindspore-lab/mindcv/blob/main/mindcv/optim/adamw.py ，则理论上 flux_controlnet 的参数精度应该也可以设为半精度。

| precision   | vae  | textencoders | flux_transformer | flux_controlnet |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | bf16         | bf16         | bf16        |
| computation | bf16 | bf16         | bf16         | bf16        |

训练精度与性能等验证结果后续待完善实验后补充讨论。