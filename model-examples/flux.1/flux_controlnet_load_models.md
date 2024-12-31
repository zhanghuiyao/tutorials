# 模型加载、controlnet 初始化与模块精度设置

延续 FLUX.1 lora dreambooth 微调 [模型加载、LoRA层初始化与模块精度设置](flux_lora_load_models.md) 中的介绍，本文继续对 FLUX.1 controlnet 微调作对应的简单的实践分享。上篇提到的一些点可能不在此重复展开，建议按顺序阅读。


## 模型加载与 controlnet 初始化

FLUX.1 controlnet 微调对比 lora 微调，需要额外定义 flux_controlnet 模型。其他部分与 flux lora 微调的模型加载一致：

```python
from transformers import AutoTokenizer, PretrainedConfig
from mindone.transformers import CLIPTextModel, T5EncoderModel
from mindone.diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel

# Load the tokenizers
tokenizer_one = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
)
# load t5 tokenizer
tokenizer_two = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=args.revision,
)
# load clip text encoder
text_encoder_one = CLIPTextModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
)
# load t5 text encoder
text_encoder_two = T5EncoderModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
)

vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
    revision=args.revision,
    variant=args.variant,
)
flux_transformer = FluxTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="transformer",
    revision=args.revision,
    variant=args.variant,
)

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="scheduler",
)
```
`FluxControlNetModel` 模型在 mindONE.diffusers 0.30 版本支持。定义时可以通过 `from_pretrained` 从社区上加载训练好的 controlnet 模型，也可以直接按照自己需要的通过 `from_transformer` 接口根据定义好的 flux_transformer 初始化一个全新的。

```python
if args.controlnet_model_name_or_path:
    logger.info("Loading existing controlnet weights")
    flux_controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
    logger.info("Initializing controlnet weights from transformer")
    # we can define the num_layers, num_single_layers,
    flux_controlnet = FluxControlNetModel.from_transformer(
        flux_transformer,
        attention_head_dim=flux_transformer.config["attention_head_dim"],
        num_attention_heads=flux_transformer.config["num_attention_heads"],
        num_layers=args.num_double_layers,
        num_single_layers=args.num_single_layers,
    )

```

加载后，我们对不需要训练的模型进行参数冻结，设 `requires_grad = False`。这里只有 `flux_controlnet` 不需要处理，默认是 `True`。


```python
# We only train the flux_controlnet
from mindspore import nn
def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False

freeze_params(vae)
freeze_params(flux_transformer)
freeze_params(text_encoder_one)
freeze_params(text_encoder_two)
```

## 模块精度设置

精度设置对我们的微调结果比较重要，这部分介绍一下 flux controlnet 微调实践时的各个模块的参数精度、运行精度的设置。假设我们只微调 transformers 的部分。首先看一下 flux_transformer 参数量大约为 11.9B，不过这部分只参与前向。参与训练的 flux_controlnet 参数量大约 1.44B。比 LoRA 微调还是来说是“重量级”训练。考虑到训练性能和显存问题，我们暂时把不参与训练的参数精度设为 `bf16`。

```python
# 查看参数量的样例代码
all_params = sum(p.numel() for p in flux_transformer.get_parameters())
trainable_params = sum(p.numel() for p in flux_controlnet.trainable_params())
```

vae, flux_transformer 不参与训练的模块，只参与前向运算，权重无需保持全精度。`flux_controlnet` 使用 to_float() 实现手动混精，使用半精度计算，其参数的精度则存为全精度 `fp32`，因为我们使用 mindspore 框架的优化器，比如 `nn.AdamWeightDecay`, 当前是按照参数的精度做梯度更新的，而不会在反向更新权重时自动 upcast。假如训练参数也设置成半精度，变成完全的半精度训练，在梯度更新时可能会导致溢出，无法正常训练。

相关代码片段以供参考：

```python
import mindspore as ms
# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = ms.float32
if args.mixed_precision == "fp16":
    weight_dtype = ms.float16
elif args.mixed_precision == "bf16":
    weight_dtype = ms.bfloat16

vae.to(dtype=weight_dtype)
flux_transformer.to(dtype=weight_dtype)

flux_controlnet.to_float(weight_dtype)
```

这里我们暂时没有处理 text encoders，huggingface diffusers 给出的训练样例是 clip 和 T5 以全精度，训练前通过 `compute_embeddings` 计算后删除以释放内存。

当前 `args.mixed_precision = bf16` 为例，预期各模块的参数精度、计算精度为：

| precision   | vae  | textencoders | flux_transformer | flux_controlnet |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | fp32         | bf16         | fp32        |
| computation | bf16 | fp32         | bf16         | bf16        |


但是参考文档 [Run 🤗Diffusers-Style Training on MindSpore](https://gist.github.com/townwish4git/3a181a1884747dfbbe4b31107ec02166)所说，目前MindSpore内存池没有清空内存碎片的功能，text encoders 载入时分配了的显存，del 之后并没有真正释放。为了抠点显存，或许可以尝试计算后 text encoders 的`compute_embeddings` 完成计算后，先转成 bf16，然后再删除（反正可能是假性删除）。

```python
# 文本计算
... = compute_embeddings(text_encoder_one, text_encoder_two)

# 算好后可以删除了，防止假性删除，删除前先转一下半精度
text_encoder_one.to(dtype=weight_dtype)
text_encoder_two.to(dtype=weight_dtype)
del text_encoder_one, text_encoder_two
```

上面的代码片段可读性差，容易让人疑惑，显存够的话可以不需要这样处理。又或者我们直接使用 `bf16` 计算 text embedding，训练精度如果没问题也ok。

| precision   | vae  | textencoders | flux_transformer | flux_controlnet |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | bf16         | bf16         | fp32        |
| computation | bf16 | bf16         | bf16         | bf16        |

```python
import mindspore as ms
# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = ms.float32
if args.mixed_precision == "fp16":
    weight_dtype = ms.float16
elif args.mixed_precision == "bf16":
    weight_dtype = ms.bfloat16

vae.to(dtype=weight_dtype)
flux_transformer.to(dtype=weight_dtype)
text_encoder_one.to(dtype=weight_dtype)
text_encoder_two.to(dtype=weight_dtype)
flux_controlnet.to_float(weight_dtype)
```
