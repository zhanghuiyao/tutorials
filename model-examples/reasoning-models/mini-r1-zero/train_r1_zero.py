import argparse
import ast

import mindspore
import numpy as np
from typing import Dict
from transformers import AutoTokenizer
from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper, auto_mixed_precision
from mindone.transformers.models.qwen2 import Qwen2ForCausalLM

import mindspore as ms
from mindspore import nn

from src.dataset import create_countdown_dataset
from src.grpo import GRPO
from src.rewards import format_reward, countdown_game_accuracy_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="pretrained model")
    parser.add_argument("--dataset-path", type=str, default="Jiayi-Pan/Countdown-Tasks-3to4", help="dataset path.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")  # FIXME
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--zero-stage", type=int, default=0, choices=[0, 1, 2], help="stage of ZeRO optimizer parallelism"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="whether or not to enable mix precision with float16"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="whether or not to enable mix precision with bfloat16"
    )
    parser.add_argument(
        "--is-distribute", type=ast.literal_eval, default=False, help="whether or not to run distribute"
    )
    parser.add_argument("--rank", type=int, default=0, help="id of card")
    parser.add_argument("--rank_size", type=int, default=1, help="num of cards")
    args = parser.parse_args()
    print(f"{args=}")

    # 0. set mindspore context
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"}, pynative_synchronize=True)
    if args.is_distribute:
        from mindspore.communication import get_group_size, get_rank, init

        init()
        args.rank = get_rank()
        args.rank_size = get_group_size()
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=get_group_size(),
        )

    # 1. create dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset, test_dataset = create_countdown_dataset(
        args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_epochs=1,
        rank=args.rank,
        rank_size=args.rank_size,
    )

    # 2. create train network and mix precision
    policy_model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        use_flash_attention_2=False,
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
        return_dict=False,
    )
    reference_model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        use_flash_attention_2=False,
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
        return_dict=False,
    )

    policy_model.gradient_checkpointing_enable()

    assert not (args.fp16 and args.bf16)
    if args.fp16:
        policy_model = auto_mixed_precision(policy_model, "O2", ms.float16)
        reference_model = auto_mixed_precision(reference_model, "O2", ms.float16)
    if args.bf16:
        policy_model = auto_mixed_precision(policy_model, "O2", ms.bfloat16)
        reference_model = auto_mixed_precision(reference_model, "O2", ms.float16)

    if args.zero_stage == 0:
        optimizer = nn.AdamWeightDecay(policy_model.trainable_params(), learning_rate=args.learning_rate)
    elif args.zero_stage == 1:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO1
        optimizer = AdamWeightDecayZeRO1(policy_model.trainable_params(), learning_rate=args.learning_rate)
    elif args.zero_stage == 2:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO2
        optimizer = AdamWeightDecayZeRO2(policy_model.trainable_params(), learning_rate=args.learning_rate)
    else:
        raise ValueError

    grpo_model = GRPO(policy_model, reference_model, [format_reward, countdown_game_accuracy_reward], tokenizer, args)

    class TrainNet(nn.Cell):
        def __init__(self, grpo_model: GRPO):
            super(TrainNet, self).__init__(auto_prefix=False)
            self.grpo_model = grpo_model

        def construct(self, *args, **kwargs):
            loss = self.grpo_model.compute_loss(*args, **kwargs)
            return loss

    train_model = TrainOneStepWrapper(TrainNet(grpo_model), optimizer)

    # 3. training
    train_model.set_train()
    for step, batch in enumerate(train_dataset):
        batch = batch["item"]
        prompt_completion_ids, num_logits_to_keep, completion_mask, rewards = \
            grpo_model.get_completion_and_reward(batch)

        loss, _, overflow = train_model(
            mindspore.Tensor(prompt_completion_ids, mindspore.int32),
            num_logits_to_keep,
            mindspore.Tensor(completion_mask, mindspore.bool_),
            mindspore.Tensor(rewards, mindspore.float32),
        )

        print(f"step: {step}, loss: {loss}")


if __name__ == "__main__":
    main()
