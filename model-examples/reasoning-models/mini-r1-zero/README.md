# Mini R1-Zero with MindSpore

*A minimal reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) with native MindSpore.*


## Tutorials

See [train_a_mini_r1_zero_from_scratch.md](./tutorials-docs/train_a_mini_r1_zero_from_scratch.md)


## Features

- [x] grpo rl method
- [x] format reward, accuracy reward(countdown)
- [x] countdown game
- [x] base model: Qwen2.5-1.5B-Instruct
- [x] trainable on Ascend* device

- [ ] (TODO) large scale training
- [ ] (TODO) evaluation
- [ ] (TODO) visualization of results
- [ ] (TODO) ai-mo math task and reward


## Installation

```shell
pip install git+https://github.com/zhanghuiyao/mindone.git@add_qwen2
```


## Run Training

```shell
python train_r1_zero.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-path Jiayi-Pan/Countdown-Tasks-3to4 \
  --max-completion-length 256 \
  --bf16 \
  --is-distribute False
```


## Acknowledge
* DeepSeek R1 [paper](https://arxiv.org/abs/2501.12948)
* DeepSeek Math [paper](https://arxiv.org/abs/2402.03300)
* We use Qwen2.5 series base model [Qwen2.5](https://github.com/QwenLM/Qwen2.5).


## Citation
```
@misc{mini-r1-zero-ms,
author       = {mindspore-lab teams},
title        = {Mini R1-Zero with MindSpore},
howpublished = {https://github.com/mindspore-lab/tutorials/model-examples/reasoning-models/mini-r1-zero},
note         = {Accessed: 2025-02-12},
year         = {2025}
}
```
