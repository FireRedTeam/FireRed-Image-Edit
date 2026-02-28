import contextlib
import torch

USE_NVTX = False

@contextlib.contextmanager
def maybe_nvtx_range(msg: str):
    if USE_NVTX:
        torch.cuda.nvtx.range_push(msg)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value