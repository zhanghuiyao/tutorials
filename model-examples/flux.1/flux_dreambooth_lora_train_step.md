# Flux.1 Dreambooth LoRA 单步训练构建

我们使用mindone.diffusers 组件封装好的 MindSpore 单步训练基本类 `TrainStep` ，定义Flux Dreambooth Lora 微调的单步训练类 `TrainStepForFluxDevDB`。自定义的单步训练类会继承基本类中的 `contruct` 方法，里面包含前向loss计算、反向梯度计算和优化器更新的全过程。我们只需要根据自己的模型定义前向计算loss的过程到 `forward` 方法即可。

## 初始化 `__init__` 实现

假设我们使用 [dog](https://huggingface.co/datasets/diffusers/dog-example) 数据集做 dreambooth lora 微调，微调训练数据集共用 prompt "a photo of sks dog"。本案例中 text encoder 不参与训练，我们可以提前算好文本嵌入，text encoders 部分不涉及梯度计算，encode 的计算过程不需要封装到 forward 的方法中。我们在该 TrainStep Cell 初始化的时候传入"a photo of sks dog"的 编码后的输出 prompt_embeds, pooled_prompt_embeds, text_ids 即可。

抽象类 `TrainStep` __init__方法有 self.model = model.set_grad(True)，操作，我们要训练的是 flux transformer 的部分，因此在super().__init__的时候传入 `model = transformer`。

另外MindSpore 静态图的写法不支持字典 dict.key 的调用方法，这里一般有两类参数字典需要在初始化处理：

- 前向计算使用到的、存在模型 config 字典里的个别参数，在初始化方法中单独取出
- 通过训练脚本传入的 args 参数字典，需要使用 `AttrJitWrapper` 转换一下， 这本质上是 dict.key -> dict[key] 的调用转换。

```python
# AttrJitWrapper 在 mindone.diffusers.training_utils 提供
@ms.jit_class
class AttrJitWrapper:
    def __init__(self, **kwargs):
        for name in kwargs:
            setattr(self, name, kwargs[name])
```
具体的处理动作如下：

```python
def __init__(.., vae, ..., args, ...):
    ...
    # 1、涉及到字典调用的模型参数，需要使用的单独取出
    self.vae_config_scaling_factor = vae.config.scaling_factor
    self.vae_config_shift_factor = vae.config.shift_factor
    self.vae_config_block_out_channels = vae.config.block_out_channels
    self.vae_scale_factor = 2 ** (len(self.vae_config_block_out_channels))
    ...

    # 2、args 参数组处理
    self.args = AttrJitWrapper(**vars(args))
    ...
```

完整的 `TrainStepForFluxDevDB` 初始化函数定义如下：


```python
from mindspore import nn
from mindspore.amp import StaticLossScaler
from mindone.diffusers.training_utils import AttrJitWrapper, TrainStep

class TrainStepForFluxDevDB(TrainStep):
    
    def __init__(
        self,
        vae: nn.Cell,
        transformer: nn.Cell,
        optimizer: nn.Optimizer,
        noise_scheduler,
        weight_dtype,
        length_of_dataloader,
        args,                   # args dicts 
        prompt_embeds,          # outputs of text encoders 
        pooled_prompt_embeds,   # outputs of text encoders 
        text_ids,               # outputs of text encoders 
    ):
        # 初始化基类 `TrainStep` __init__方法， model 传入要微调的模型
        super().__init__(
            model = transformer,
            optimizer = optimizer,
            loss_scaler = StaticLossScaler(4096),
            max_grad_norm = args.max_grad_norm,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            gradient_accumulation_kwargs=dict(length_of_dataloader=length_of_dataloader),
        )
        # transformer
        self.transformer = transformer
        self.transformer_config_guidance_embeds = transformer.config.guidance_embeds

        # vae
        self.vae = vae
        self.vae_dtype = vae.dtype
        self.vae_config_scaling_factor = vae.config.scaling_factor
        self.vae_config_shift_factor = vae.config.shift_factor
        self.vae_config_block_out_channels = vae.config.block_out_channels
        self.vae_scale_factor = 2 ** (len(self.vae_config_block_out_channels))

        # noise_scheduler and others args
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.weight_dtype = weight_dtype
        self.args = AttrJitWrapper(**vars(args))

        # 提前计算好的 text embedings
        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.text_ids = text_ids
```

## forward() 实现

整个前向方法基本上与 huggingface diffusers (version=0.30) 前向计算的定义流程保持一致。
注意点：这里重写了 `get_sigmas` 方法，因为原来的写法在 mindspore 图模式不支持，而重写的版本可以在图模式跑通，同时通过直接从 indice 获取 sigma 调用更少的 ops 算子。 guidance 的 torch.Tensor.expand 操作使用 mindspore.Tensor.broadcast_to 替代。

```python

# before
def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# after
class TrainStepForFluxDevDB(TrainStep):
    ...
    def get_sigmas(self, indices, n_dim=4, dtype=ms.float32):
        """
        origin `get_sigmas` which uses timesteps to get sigmas might be not supported
        in mindspore Graph mode, thus we rewrite `get_sigmas` to get sigma directly
        from indices which calls less ops and could run in mindspore Graph mode.
        """
        sigma = self.noise_scheduler.sigmas[indices].to(dtype=dtype)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

# before
guidance = guidance.expand(model_input.shape[0])
# after
guidance = guidance.broadcast_to((model_input.shape[0],))

```

完整的 forward 方法定义如下：


```python
import mindspore as ms 
from mindspore import ops, nn
from mindone.diffusers import FluxPipeline
from mindone.diffusers.training_utils import compute_loss_weighting_for_sd3

def compute_weighting_mse_loss(weighting, pred, target):
    """
    When argument with_prior_preservation is True in DreamBooth training, weighting has different batch_size
    with pred/target which causes errors, therefore we broadcast them to proper shape before mul
    """
    repeats = weighting.shape[0] // pred.shape[0]
    target_ndim = target.ndim
    square_loss = ((pred.float() - target.float()) ** 2).tile((repeats,) + (1,) * (target_ndim - 1))

    weighting_mse_loss = ops.mean(
        (weighting * square_loss).reshape(target.shape[0], -1),
        1,
    )
    weighting_mse_loss = weighting_mse_loss.mean()

    return weighting_mse_loss


class TrainStepForFluxDevDB(TrainStep):
    
    def __init__(...):
        ...

    def get_sigmas(self, indices, n_dim=4, dtype=ms.float32):
        sigma = self.noise_scheduler.sigmas[indices].to(dtype=dtype)
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def forward(self, pixel_values):
            
            # use pre-computed embeddings.
            prompt_embeds, pooled_prompt_embeds, text_ids = (
                self.prompt_embeds,
                self.pooled_prompt_embeds,
                self.text_ids,
            )

            # Convert images to latent space
            pixel_values = pixel_values.to(dtype=self.vae_dtype)

            model_input = self.vae.diag_gauss_dist.sample(self.vae.encode(pixel_values)[0])
            model_input = (model_input - self.vae_config_shift_factor) * self.vae_config_scaling_factor
            model_input = model_input.to(dtype=self.weight_dtype)

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2],
                model_input.shape[3],
                self.weight_dtype,
            )

            # Sample noise that we'll add to the latents
            noise = ops.randn_like(model_input, dtype=model_input.dtype)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            if self.args.weighting_scheme == "logit_normal":
                # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                u = ops.normal(mean=self.args.logit_mean, stddev=self.args.logit_std, shape=(bsz,))
                u = ops.sigmoid(u)
            elif self.args.weighting_scheme == "mode":
                u = ops.rand(bsz)
                u = 1 - u - self.args.mode_scale * (ops.cos(ms.numpy.pi * u / 2) ** 2 - 1 + u)
            else:
                u = ops.rand(bsz)

            indices = (u * self.noise_scheduler_num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices]

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = self.get_sigmas(indices, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

            packed_noisy_model_input = FluxPipeline._pack_latents(
                noisy_model_input,
                batch_size=model_input.shape[0],
                num_channels_latents=model_input.shape[1],
                height=model_input.shape[2],
                width=model_input.shape[3],
            )

            # handle guidance
            if self.transformer_config_guidance_embeds:
                guidance = ms.tensor([self.args.guidance_scale])
                guidance = guidance.broadcast_to((model_input.shape[0],))
            else:
                guidance = None

            # Predict the noise residual
            model_pred = self.transformer(
                hidden_states=packed_noisy_model_input,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing) # noqa E501
                timestep=timesteps / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=int(model_input.shape[2] * self.vae_scale_factor / 2),
                width=int(model_input.shape[3] * self.vae_scale_factor / 2),
                vae_scale_factor=self.vae_scale_factor,
            )

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.args.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            target = noise - model_input

            if self.args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = ops.chunk(model_pred, 2, axis=0)
                target, target_prior = ops.chunk(target, 2, axis=0)

                # Compute prior loss
                prior_loss = compute_weighting_mse_loss(weighting, model_pred_prior, target_prior)

            # Compute regular loss.
            loss = compute_weighting_mse_loss(weighting, model_pred, target)

            if args.with_prior_preservation:
                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss

            loss = self.scale_loss(loss)
            return loss, model_pred
```
