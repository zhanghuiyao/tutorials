import numpy as np

from transformers import PreTrainedTokenizer
from datasets import load_dataset
from mindone.transformers.mindspore_adapter.data import HF2MSDataset

import mindspore


def create_countdown_dataset(
        hf_dataset_path: str = "Jiayi-Pan/Countdown-Tasks-3to4",
        tokenizer: PreTrainedTokenizer = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        rank: int = 0,
        rank_size: int = 1,
):

    # generate r1 prompt with a prefix for the model to already start with the thinking process

    # TODO: remove <think>
    def generate_r1_prompt(numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
        },
            {
                "role": "user",
                "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }]
        return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
                "target": target}

    # Load dataset from Hugging Face Hub
    dataset = load_dataset(hf_dataset_path, split="train")
    # select a random subset of 50k samples
    dataset = dataset.shuffle(seed=42).select(range(50000))

    # convert our dataset to the r1 prompt
    dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

    # split the dataset into train and test
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # convert hf-datasets to mindspore-dataset

    def ms_data_collator(inputs, batch_info):
        first = inputs[0]
        assert isinstance(first, dict)
        prompts = [x["prompt"] for x in inputs]
        nums = [x["nums"] for x in inputs]
        targets = np.array([int(x["target"]) for x in inputs])
        # prompt_inputs = tokenizer(prompts, return_tensors="np", padding=True, padding_side="left", add_special_tokens=False)
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=256,
            padding_side="right",
            add_special_tokens=False
        )
        batch = {
            "prompts": prompts,
            "nums": nums,
            "targets": targets,
            "prompt_ids": prompt_inputs.input_ids,
            "attention_mask": prompt_inputs.attention_mask,
        }
        return batch

    ms_train_dataset = mindspore.dataset.GeneratorDataset(
        HF2MSDataset(train_dataset), column_names="item", shard_id=rank, num_shards=rank_size
    )
    ms_train_dataset = ms_train_dataset.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
    ms_train_dataset = ms_train_dataset.repeat(1)
    ms_train_dataset = ms_train_dataset.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)

    ms_test_dataset = mindspore.dataset.GeneratorDataset(
        HF2MSDataset(test_dataset), column_names="item", shard_id=rank, num_shards=rank_size
    )
    ms_test_dataset = ms_test_dataset.batch(batch_size=1, per_batch_map=ms_data_collator)
    ms_test_dataset = ms_test_dataset.repeat(1)
    ms_test_dataset = ms_test_dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    return ms_train_dataset, ms_test_dataset
