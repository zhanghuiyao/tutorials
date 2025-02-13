import numpy as np
from typing import Callable, Optional, List
from transformers import PreTrainedTokenizer, GenerationConfig

import mindspore as ms
from mindspore import nn, ops, Tensor


class GRPO(nn.Cell):
    def __init__(
            self,
            policy_model: Optional[nn.Cell],
            reference_model: Optional[nn.Cell],
            reward_funcs: List[Callable],
            tokenizer: PreTrainedTokenizer,
            args
    ):
        super(GRPO, self).__init__()

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_funcs = reward_funcs
        self.tokenizer = tokenizer

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations              # = G in the GRPO paper
        self.beta = args.beta

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_generations,
            pad_token_id=tokenizer.pad_token_id,
        )

    def set_policy_train(self):

        # 0. setting models' training mode
        self.policy_model.set_train(True)
        self.reference_model.set_train(False)
        for rm in self.reward_funcs:
            if isinstance(rm, nn.Cell):
                rm.set_train(False)

    def get_completion_and_reward(self, batch):
        prompts, nums, targets, prompt_ids, attention_mask = \
            batch["prompts"], batch["nums"], batch["targets"], batch["prompt_ids"], batch["attention_mask"]
        prompts = np.array(prompts).tolist()
        nums = [np.array(b).tolist() for b in batch["nums"]]

        # FIXME: unpad
        assert prompt_ids.shape[0] == 1, "not support bs>1 when generate task"
        prompt_ids = prompt_ids[:, :attention_mask.sum()]
        attention_mask = attention_mask[:, :attention_mask.sum()]

        completion_ids = self.policy_model.generate(
            input_ids=Tensor(prompt_ids, ms.int32),
            attention_mask=Tensor(attention_mask, ms.bool_),
            generation_config=self.generation_config,
            max_new_tokens=self.max_completion_length,
            use_cache=False,
        )
        completion_ids = completion_ids.asnumpy()
        prompt_completion_ids = np.concatenate([prompt_ids.repeat(self.num_generations, axis=0), completion_ids], axis=-1)
        num_logits_to_keep = completion_ids.shape[1]

        # Mask everything after the first EOS token
        is_eos = np.array(completion_ids == self.tokenizer.eos_token_id)
        eos_idx = np.full((is_eos.shape[0],), is_eos.shape[1], dtype=np.int)
        eos_idx[is_eos.any(axis=1)] = is_eos.astype(np.int).argmax(axis=1)[is_eos.any(axis=1)]
        sequence_indices = np.arange(is_eos.shape[1])[None].repeat(is_eos.shape[0], axis=0)
        completion_mask = (sequence_indices <= eos_idx[:, None]).astype(np.int)

        # get reward
        # decode the generated completions
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = np.zeros((len(prompts), len(self.reward_funcs)), dtype=np.float32)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards_per_func[:, i] = reward_func(completions=completions, nums=nums, targets=targets, prompt=prompts)  # Shape (B*G,)
        rewards = rewards_per_func.sum(axis=1)

        return prompt_completion_ids, num_logits_to_keep, completion_mask, rewards

    def compute_loss(
            self,
            prompt_completion_ids: Tensor,
            num_logits_to_keep: int,
            completion_mask: Tensor,
            rewards: Tensor
    ) -> Tensor:

        # 1. compute grpo reward and advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(axis=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(axis=1)

        # normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 2. compute kl divergence
        per_token_logps = self.get_log_probabilities(
            prompt_completion_ids,
            self.policy_model(prompt_completion_ids)[0],
            num_logits_to_keep,
        )
        ref_per_token_logps = self.get_log_probabilities(
            prompt_completion_ids,
            self.reference_model(prompt_completion_ids)[0],
            num_logits_to_keep,
        )
        kl_divergence = ops.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = ops.exp(per_token_logps - ops.stop_gradient(per_token_logps)) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * kl_divergence)
        loss = ((per_token_loss * completion_mask).sum(axis=1) / completion_mask.sum(axis=1)).mean()

        return loss

    @ms.jit
    def get_log_probabilities(self, input_ids: Tensor, logits: Tensor, num_logits_to_keep: int) -> Tensor:
        # logits: (B, L, V), prompt_completion_logits -> completion_logits
        logits = logits[:, -(num_logits_to_keep+1):-1, :]

        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = ()

        for i in range(logits.shape[0]):
            logits_row, input_ids_row = logits[i], input_ids[i, -num_logits_to_keep:]

            log_probs = ops.log_softmax(logits_row, axis=-1)
            token_log_prob = ops.gather_elements(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps += (token_log_prob,)

        return ops.stack(per_token_logps)




