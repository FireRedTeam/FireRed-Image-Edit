# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
import contextlib
import torch
import time

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


def timer(func):
    def wrapper(*args, **kwargs):
        start_time  = time.time()
        result      = func(*args, **kwargs)
        end_time    = time.time()
        print(f"function {func.__name__} running for {end_time - start_time} seconds")
        return result
    return wrapper

def timer_record(model_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start_time = time.time()
            result      = func(*args, **kwargs)
            torch.cuda.synchronize()
            end_time = time.time()
            import torch.distributed as dist
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    time_sum  = end_time - start_time
                    print('# --------------------------------------------------------- #')
                    print(f'#   {model_name} time: {time_sum}s')
                    print('# --------------------------------------------------------- #')
                    _write_to_excel(model_name, time_sum)
            else:
                time_sum  = end_time - start_time
                print('# --------------------------------------------------------- #')
                print(f'#   {model_name} time: {time_sum}s')
                print('# --------------------------------------------------------- #')
                _write_to_excel(model_name, time_sum)
            return result
        return wrapper
    return decorator

def _write_to_excel(model_name, time_sum):
    import os

    import pandas as pd

    row_env = os.environ.get(f"{model_name}_EXCEL_ROW", "1")  # 默认第1行
    col_env = os.environ.get(f"{model_name}_EXCEL_COL", "1")  # 默认第A列
    file_path = os.environ.get("EXCEL_FILE", "timing_records.xlsx")  # 默认文件名

    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1", header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    row_idx = int(row_env)
    col_idx = int(col_env)

    if row_idx >= len(df):
        df = pd.concat([df, pd.DataFrame([ [None] * (len(df.columns) if not df.empty else 0) ] * (row_idx - len(df) + 1))], ignore_index=True)

    if col_idx >= len(df.columns):
        df = pd.concat([df, pd.DataFrame(columns=range(len(df.columns), col_idx + 1))], axis=1)

    df.iloc[row_idx, col_idx] = time_sum

    df.to_excel(file_path, index=False, header=False, sheet_name="Sheet1")

def get_autocast_dtype():
    try:
        if not torch.cuda.is_available():
            print("CUDA not available, using float16 by default.")
            return torch.float16

        device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(device)

        print(f"GPU: {prop.name}, Compute Capability: {prop.major}.{prop.minor}")

        if prop.major >= 8:
            if torch.cuda.is_bf16_supported():
                print("Using bfloat16.")
                return torch.bfloat16
            else:
                print("Compute capability >= 8.0 but bfloat16 not supported, falling back to float16.")
                return torch.float16
        else:
            print("GPU does not support bfloat16 natively, using float16.")
            return torch.float16

    except Exception as e:
        print(f"Error detecting GPU capability: {e}, falling back to float16.")
        return torch.float16