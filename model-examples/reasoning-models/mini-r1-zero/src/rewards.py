# reference to
# https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
# https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb
import re

import numpy as np


def format_reward(completions: list[str], *args, **kwargs) -> np.ndarray:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    matches = [re.match(pattern, "<think>" + content, re.DOTALL | re.MULTILINE) for content in completions]
    return np.array([1.0 if match else 0.0 for match in matches])


def countdown_game_accuracy_reward(completions: list[str], nums: int, targets: int, *args, **kwargs) -> np.ndarray:
    """
    For Countdown Game, evaluates completions based on: Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        targets (int): Expected answers
        nums (int): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, targets, nums):
        try:
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builti'ns__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
    return np.array(rewards)
