import torch
import nvtx
import functools
import inspect
from nvtx import annotate as nvtx_annotate
import time
from .torch_dynamo import explain


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
    """Return runtime in seconds"""
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
