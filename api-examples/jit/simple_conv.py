import time
import numpy as np
from typing import Optional

import mindspore
from mindspore import ops, nn, Tensor


x = Tensor(np.random.randn(1, 128, 256, 256), mindspore.float32)


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(
        self,
        in_channels: int = 128,
        channels: int = 128,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out

def run_func(block: nn.Cell, des:str = "function"):
    s_time = time.time()

    out = block(x)

    time_to_compile = time.time() - s_time
    s_time = time.time()

    for _ in range(10):
        out = block(x)

    time_to_run_ten_times = time.time() - s_time

    print(f"{des}, \
          output shape is: {out.shape}, \
          time to compile: {time_to_compile:.2f}s, \
          time to run thousand times: {time_to_run_ten_times:.2f}s, \
          time end to end(a thousand times): {time_to_compile+time_to_run_ten_times:.2f}")


block = BasicBlock()

run_func(block, des="origin block")

block.construct = mindspore.jit(block.construct)

run_func(block, des="jitted block by default")

