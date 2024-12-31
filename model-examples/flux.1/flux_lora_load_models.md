# 模型加载、LoRA层初始化与模块精度设置

本文介绍 Diffsuers-style FLUX.1 Dreambooth LoRA 微调开发实践中的模型定义、预训练权重加载、LoRA层初始化与模块精度部分，分享开发时可能需要注意的要点。


## 模型加载

模型的加载可直接调用 MindONE.diffusers 中已经定义好的模型接口，调用过程几乎无需关心 huggingface diffusers 与 MindONE.diffusers 模型定义过程中的区别。如果你之前没有用过 diffusers 工具，则可以继续看本节是如何定义模型的。

FLUX.1 微调需要加载的模型/模块有：
- 文本编码：CLIP & T5
- 图像编解码：vae 
- 噪声预测 ：flux_transformers
- scheduler：FlowMatchEulerDiscreteScheduler


MindONE.diffusers 支持直接加载 safetensoer 格式的模型，当然 MindSpore 框架本身也是支持 safetensors 权重的加载与保存的，详情可以查看接口[load_checkpoint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.load_checkpoint.html?highlight=load_checkpoint#mindspore.load_checkpoint), [save_checkpoint](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.save_checkpoint.html?highlight=save_checkpoint#mindspore.save_checkpoint) 的入参 `format`。 
 
脚本的传参 `args.pretrained_model_name_or_path` 传参可以直接传 huggingface 社区上的模型的 model name, `from_pretrained` 接口会自动从 hf 社区下载配置文件与权重到缓存路径。 或者提前下载好一套权重，按照指定格式摆放，传参时传本地权重路径。这里我们使用 [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)。

文本编码部分，tokenizers 直接从 `transformers` 库调用。text encoders 模型部分，我们在 `MindONE.transformers` 组件对大部分 diffusers 使用到的文本编码模型做了 mindspore 适配，此处因此用到的模型直接从 `MindONE.transformers` 加载。

```python
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from mindone.transformers import CLIPTextModel, T5EncoderModel

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        return CLIPTextModel
    
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


# Load the tokenizers
tokenizer_one = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
)
tokenizer_two = T5TokenizerFast.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=args.revision,
)


# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)

# load text encoders
text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
```

vae、flux_transformers、noise_scheduler 直接从 `MindONE.diffusers` 调用：


```python
from mindone.diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

# Load scheduler and models
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="scheduler"
)

vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
    revision=args.revision,
    variant=args.variant,
)
transformer = FluxTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
)
```

通常模型初始化后根据各个组件是否需要训练来设置各个组件中参数的 `param.requires_grad` 值，我们只微调 transformer 注入的 lora 层，所以先把上面定义好的所有模型组件的参数设置 `requires_grad = False`。MindSpore Cell 暂时没有`requires_grad_`接口，我们可以通过下面方法在训练脚本中等价实现参数冻结。


```python
# We only train the additional adapter LoRA layers
from mindspore import nn
def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False

freeze_params(transformer)
freeze_params(vae)
freeze_params(text_encoder_one)
freeze_params(text_encoder_two)
```

## LoRA 层初始化

mindONE.diffusers 集成了 🤗PEFT (Parameter-Efficient Fine-Tuning) 库，使得模型可以方便地注入微调层。mindONE.diffusers 每个具体模型继承的抽象类之一是 `PeftAdapterMixin` ，它包含用于加载和使用 PEFT 库中支持的 adapters weights 的所有函数，其中就包括 LoRA 层注入方法。例如我们要做 LoRA 微调的模型 `FluxTransformer2DModel`，定义示例如下：


```python
class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    ...
```

以下是我们通过调用 `PeftAdapterMixin` 的 `add_adapter` 方法为微调模型注入 LoRA 层的样例代码。我们还可以通过 `LoraConfig` 的 `target_modules` 指定具体微调注入的模块、指定 LoRA 的秩以及 LoRA 层权重初始化的方法。如果我们还打算 transformer + text encoder 一起微调，也可以给 text encoder （这里指的是 clip 而不是 T5）注入。

LoRA 层初始化后不需要如同上面的其他模块一般对相关 `param.requires_grad` 做处理，默认是 `True`。

```python
from mindone.diffusers._peft import LoraConfig

# now we will add new LoRA weights to the attention layers
transformer_lora_config = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_rank,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
transformer.add_adapter(transformer_lora_config)

if args.train_text_encoder:
    text_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    text_encoder_one.add_adapter(text_lora_config)

```


## 模块精度设置

精度设置对我们的微调结果比较重要，这部分介绍一下 flux lora 微调实践时的各个模块的参数精度、运行精度的设置过程以及原因。假设我们只微调 transformers 的部分。首先看一下参数量 11.91B，其中参与训练的 LoRA 层参数量大约 123.36M。考虑到训练性能和显存问题，我们没必要把所有模块的参数精度都设置为全精度。

```python
# 查看参数量的样例代码
all_params = sum(p.numel() for p in transformer.get_parameters())
trainable_params = sum(p.numel() for p in transformer.trainable_params())
```

vae, text_encoder and transformer 参与训练的模块，只参与前向运算，权重无需保持全精度。对于 LoRA 层，我们可以暂时使用 to_float() 实现手动混精，使用半精度计算，但是需要保证其参数的精度是全精度，因为我们使用 mindspore 框架的优化器，比如 `nn.AdamWeightDecay`, 当前是按照参数的精度做梯度更新的，而不会在反向更新权重时自动 upcast。假如训练参数也设置成半精度，变成完全的半精度训练，在梯度更新时可能会导致溢出，无法正常训练。

后续框架和模型应该会持续优化出更易用的写法去对标 Accelerate 提供的对应混精训练功能。

以 `args.mixed_precision = bf16` 为例，各模块的参数精度、计算精度设置如下：

| precision   | vae  | textencoders | transformers | LoRA layers |
| :---------: | :--: | :----------: | :----------: | :---------: |
| parameters  | bf16 | bf16         | bf16         | fp32        |
| computation | bf16 | bf16         | bf16         | bf16        |


相关代码片段以供参考：

```python
import mindspore as ms
from mindone.diffusers.training_utils import cast_training_params

# For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
# as these weights are only used for inference, keeping weights in full precision is not required.
weight_dtype = ms.float32
if args.mixed_precision == "fp16":
    weight_dtype = ms.float16
elif args.mixed_precision == "bf16":
    weight_dtype = ms.bfloat16

vae.to(dtype=weight_dtype)
transformer.to(dtype=weight_dtype)
text_encoder_one.to(dtype=weight_dtype)
text_encoder_two.to(dtype=weight_dtype)

models = [transformer]
if args.train_text_encoder:
    models.extend([text_encoder_one])

# Make sure the trainable params are in float32.
if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
    # only upcast trainable parameters (LoRA) into fp32
    cast_training_params(models, dtype=ms.float32)

# Prepare everything with our `accelerator`.
# LoRA layer .to_float(weight_dtype)
for peft_model in models:
    for _, module in peft_model.cells_and_names():
        if isinstance(module, BaseTunerLayer):
            for layer_name in module.adapter_layer_names:
                module_dict = getattr(module, layer_name)
                for key, layer in module_dict.items():
                    if key in module.active_adapters and isinstance(layer, nn.Cell):
                        layer.to_float(weight_dtype)
```

注意到我们在转换模型精度时使用了 `.to(dtype)`。事实上 MindSpore Cell 暂时没有 `.to(dtype)` 接口，这是 `mindONE.diffusers` 做 MindSpore 兼容时，通过模型的基本类 `ModelMixin` 手动实现的 `to` 方法，以下示例代码以供参考。


```python
class ModelMixin(nn.Cell, ...):
    r"""
    Base class for all models.
    ...
    """
    ...
    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

# ALl models base on `ModelMixin`
class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    ...
```

## 扩展阅读
- [MindSpore 自动混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/mixed_precision.html)
- [flux 模型加载、Controlnet初始化与模块精度设置](flux_controlnet_load_models.md)

