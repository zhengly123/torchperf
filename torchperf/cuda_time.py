import torch
import nvtx
import functools
import inspect
import datetime
from nvtx import annotate as nvtx_annotate
import time
from .torch_dynamo import explain
from typing import Iterable, Optional, Callable, Any


def add_nvtx_range(func):
    def wrapper(*args, **kwargs):
        with nvtx.annotate(args[0].profiling_name):
            return func(*args, **kwargs)

    return wrapper


def add_layer_name(model: torch.nn.Module, modules_to_be_hooked):
    # modules_to_be_hooked = (ResnetBlock2D,Transformer2DModel,CrossAttnUpBlock2D,UNetMidBlock2DCrossAttn,DownBlock2D,CrossAttnDownBlock2D,UpBlock2D,)
    for name, layer in model.named_modules():
        if isinstance(layer, modules_to_be_hooked):
            layer.profiling_name = name


# # Ingored in torch.compile
# def hook_model_with_nvtx(model: torch.nn.Module):
#     modules_to_be_hooked = (
#         ResnetBlock2D,
#         Transformer2DModel,
#         CrossAttnUpBlock2D,
#         UNetMidBlock2DCrossAttn,
#         DownBlock2D,
#         CrossAttnDownBlock2D,
#         UpBlock2D,
#     )
#     for name, layer in model.named_modules():
#         if isinstance(layer, modules_to_be_hooked):
#             layer.register_forward_pre_hook(partial(hook_nvtx_push, "unet." + name))
#     for name, layer in model.named_modules():
#         if isinstance(layer, modules_to_be_hooked):
#             layer.register_forward_hook(hook_nvtx_pop)


def add_nvtx_range_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with nvtx.annotate(args[0].profiling_name):
            # TODO: how to avoid [1:] here.
            return func(*args[1:], **kwargs)

    if inspect.ismethod(func):
        return wrapper.__get__(func.__self__, None)
    else:
        return wrapper


def add_perf_info(
    model: torch.nn.Module,
    modules_to_be_hooked: tuple[torch.nn.Module] = None,
    name_prefix: str = None,
    max_depth: int = None,
):
    for name, layer in model.named_modules():
        if modules_to_be_hooked is not None and not isinstance(
            layer, modules_to_be_hooked
        ):
            continue
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        depth = len(name.split("."))
        if max_depth is not None and depth > max_depth:
            continue
        layer.profiling_name = name
        layer.forward = add_nvtx_range_decorator(layer.forward)
    return model


def cuda_timeit(func, warmup=5, iters=100, compile=False, dynamic=True) -> float:
    """Return runtime in seconds"""
    return cuda_timeit_ms(func, warmup, iters, compile, dynamic) / 1000


def cuda_timeit_ms(func, warmup=5, iters=100, compile=False, dynamic=True) -> float:
    """Return runtime in milliseconds"""
    if compile:
        func = torch.compile(func, dynamic=dynamic)
    return cuda_timeit_event(func, warmup, iters)


def cuda_timeit_compile(func, iters) -> float:
    @torch.compile()
    def compiled():
        for i in range(iters):
            func()

    # (
    #     explanation,
    #     out_guards,
    #     graphs,
    #     ops_per_graph,
    #     break_reasons,
    #     explanation_verbose,
    # ) = torch._dynamo.explain(compiled)
    # if len(graphs) != 1:
    #     print(f"Warning: {len(graphs)} graphs are generated")
    # explain(compiled)

    # Warm up
    compiled()
    torch.cuda.synchronize()
    st = time.time()
    compiled()
    torch.cuda.synchronize()
    ed = time.time()
    return (ed - st) / iters


def cuda_timeit_eager(func, warmup=5, iters=100) -> float:
    """Return runtime in seconds"""
    for i in range(warmup):
        func()
    torch.cuda.synchronize()
    st = time.time()
    for i in range(iters):
        func()
    torch.cuda.synchronize()
    ed = time.time()
    return (ed - st) / iters


def cuda_timeit_event(func, warmup=5, iters=100) -> float:
    """Return runtime in milliseconds"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(warmup):
        func()
    start.record()
    for i in range(iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cuda_timeit_host_ms(func, warmup=5, iters=100) -> float:
    """Return runtime in milliseconds"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i in range(warmup):
        func()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for i in range(iters):
        func()
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    return float(t1 - t0) / 1e6 / iters


def enable_torch_profiler(name):
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f'./log/{name}-{datetime.datetime.now().strftime("%m%d-%H%M%S")}'
        ),
        # record_shapes=True,
        # with_stack=True,
        # profile_memory=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    )

    def print_and_exit(*args, **kwargs):
        prof.__exit__(prof, *args, **kwargs)

    # prof.__exit__ = print_and_exit # TODO. fix it
    return prof


def torch_profile_it(
    name,
    fn,
    *,
    warmup=1,
    with_stack=False,
    sort_keys=["self_cuda_time_total", "cuda_time_total"],
):
    torch.cuda.synchronize()
    log_name = f'./log/{name}-{datetime.datetime.now().strftime("%m%d-%H%M%S")}'
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_name),
        # record_shapes=True,
        # with_stack=with_stack,
        # profile_memory=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for i in range(warmup + 1):
            fn()
            torch.cuda.synchronize()
            prof.step()
    print(f"Writting log to {log_name}")
    for k in sort_keys:
        print(f"Sorted by {k}:")
        print(prof.key_averages().table(sort_by=k, row_limit=50))

    return prof


def torch_profile_decorator(name):
    assert isinstance(name, str)

    def decorator(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            log_name = f'./log/{name}-{datetime.datetime.now().strftime("%m%d-%H%M%S")}'
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_name),
                # record_shapes=True,
                # with_stack=True,
                # profile_memory=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                for i in range(2):
                    result = fn(*args, **kwargs)
                    torch.cuda.synchronize()
                    prof.step()
            print(f"Writting log to {log_name}")
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
            )
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            exit()
            return result

        return wrapper

    return decorator


class profile_with_sync(torch.profiler.profile):
    def __init__(
        self,
        log_name: str = "unnamed",
        cuda_sync: bool = True,
        print_sort_by: str | Iterable[str] = [
            "cuda_time_total",
            "self_cuda_time_total",
        ],
        *,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        schedule=None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config=None,
        execution_trace_observer=None,
    ):
        if on_trace_ready is None and log_name is not None:
            on_trace_ready = torch.profiler.tensorboard_trace_handler(log_name)
        self.log_name = log_name
        self.cuda_sync = cuda_sync
        self.print_sort_by = (
            [print_sort_by] if isinstance(print_sort_by, str) else print_sort_by
        )
        super().__init__(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            experimental_config=experimental_config,
            execution_trace_observer=execution_trace_observer,
        )

    def __enter__(self):
        if self.cuda_sync == True:
            torch.cuda.synchronize()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cuda_sync == True:
            torch.cuda.synchronize()
        # self.step()
        super().__exit__(exc_type, exc_val, exc_tb)
        print(f"Writting log to {self.log_name}")
        for sort_by in self.print_sort_by:
            print(self.key_averages().table(sort_by=sort_by, row_limit=50))


class time_with_sync:
    def __init__(self, verbose: str | bool | None = None, cuda_sync: bool = True):
        self.verbose = verbose
        self.cuda_sync = cuda_sync

    def __enter__(self):
        torch.cuda.synchronize()
        if self.cuda_sync == True:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cuda_sync == True:
            torch.cuda.synchronize()
        self.t1 = time.perf_counter_ns()
        self.time_ms = float(self.t1 - self.t0) / 1e6 / 1
        if isinstance(self.verbose, str):
            print(f"{self.verbose} {self.time_ms:.1f} ms")
        elif self.verbose == True:
            print(f"{self.time_ms:.1f} ms")
