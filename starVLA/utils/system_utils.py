import time
import os
import torch
import torch.distributed as dist


def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0): print(s)


class Timing:
    """
    From https://github.com/sxyu/svox2/blob/ee80e2c4df8f29a407fda5729a494be94ccf9234/svox2/utils.py#L611
    
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")

class Timer:
    def __init__(self, name, print_fn=print):
        self.name = name
        self.print_fn = print_fn
        self.is_local_rank0 = self._is_local_rank0()

    def _is_local_rank0(self):
        # accelerate / torchrun / deepspeed 通用
        try:
            return int(os.environ.get("LOCAL_RANK", 0)) == 0
        except ValueError:
            return True

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        if self.is_local_rank0:
            self.print_fn(f"{self.name} elapsed {elapsed_ms:.3f} ms")