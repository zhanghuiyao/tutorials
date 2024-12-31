# Diffuser-Style FLUX.1 training MindSpore 实践

Diffusers是一个用于生成图像、音频甚至3D结构的最先进的预训练扩散模型库。[MindONE](https://github.com/mindspore-lab/mindone)仓立足“Run Diffusers on MindSpore”的理念，基于MindSpore全场景AI框架实现了原生Diffusers库里包括预训练模型、噪声调度器和扩散管道在内的几乎所有组件和接口。依赖这些组件和接口，MindONE对齐 Diffusers 原仓提供了模型构建、扩散调度和快速推理等功能。

本目录下的文章介绍了如何基于MindSpore框架以及 [MindONE.diffusers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers#readme) 组件，实现当前SOTA的文生图 Flux.1 dev的微调训练。

训练中使用到的模型均直接从[MindONE.diffusers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers#readme) 组件加载。整个 diffusers-style 训练流程的开发参考了文档 [Run 🤗Diffusers-Style Training on MindSpore](https://gist.github.com/townwish4git/3a181a1884747dfbbe4b31107ec02166)，对 FLUX.1-dev 模型 dreambooth-lora 、controlnet 微调方法作在 mindspore 框架下做对应的具体实践。建议先阅读本自然段给出跳转链接的 2 篇文档。

【本系列持续更新中...】

## 单步训练的抽象类的使用介绍
mindone 仓 [example/diffusers](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers) 下提供的一系列 diffusers-style 训练脚本样例，具体的单步训练实现时都依赖这个抽象类。
- [x] [`TrainStep` in MindONE.diffusers](trainstep_in_mindone_diffusers.md)


## Dreambooth LoRA 微调流程构建介绍
完整训练脚本已上库，可参考 [🔗](https://github.com/mindspore-lab/mindone/blob/master/examples/diffusers/dreambooth/README_flux.md)
- [x] [模型加载、LoRA层初始化与模块精度设置](flux_lora_load_models.md)
- [ ] Dreambooth LoRA 训练数据集构建
- [ ] 优化器、优化参数等设置
- [x] [Flux Dreambooth LoRA 单步训练实现](flux_dreambooth_lora_train_step.md)
- [ ] enable textencoder training VS not 实现


## Controlnet 微调流程构建介绍
完整训练脚本待上库，稍后补充🔗
- [x] [模型加载、Controlnet层初始化与各模块精度设置](flux_controlnet_load_models.md)
- [ ] [尝试更换可 upcast 精度的优化器](flux_controlnet_precison_setting.md)
- [ ] Flux Controlnet 单步训练实现

## 扩展阅读

关键模型结构简介与 MindSpore 代码实现走读:
- [ ] `FluxTransformer2DModel` 实现
- [ ] `FluxControlNetModel` 实现

其他问题：
- [x] [手动下载预训练模型配置文件时的小坑](mannual_download_models_faq.md)