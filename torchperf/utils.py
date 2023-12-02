import torch
from typing import Iterable


def tensors_to_shapes(tensors):
    if isinstance(tensors, torch.Tensor):
        return tensors.shape
    elif isinstance(tensors, list):
        return [tensors_to_shapes(t) for t in tensors]
    elif isinstance(tensors, tuple):
        return tuple(tensors_to_shapes(t) for t in tensors)
    elif isinstance(tensors, dict):
        return {k: tensors_to_shapes(v) for k, v in tensors.items()}
    elif not isinstance(tensors, Iterable):
        return tensors
    else:
        raise ValueError(f"Unknown type {type(tensors)}")


def shapes_to_tensors(shapes, old_batch=None, new_batch=None):
    if isinstance(shapes, torch.Size):
        if len(shapes) == 0:  # Timestamp
            print(f"Find 0-dim tensor and replace it with 1")
            return torch.tensor(1)
        if old_batch is not None:
            assert shapes[0] == old_batch
        if new_batch is None:
            new_batch = list(shapes)[0]
        return torch.randn(
            [new_batch] + list(shapes)[1:]  # , dtype=torch.float16, device="cuda"
        )
    elif isinstance(shapes, list):
        return [shapes_to_tensors(t, old_batch, new_batch) for t in shapes]
    elif isinstance(shapes, tuple):
        return tuple(shapes_to_tensors(t, old_batch, new_batch) for t in shapes)
    elif isinstance(shapes, dict):
        return {
            k: shapes_to_tensors(v, old_batch, new_batch) for k, v in shapes.items()
        }
    elif not isinstance(shapes, Iterable):
        return shapes
    else:
        raise ValueError(f"Unknown type {type(shapes)}")


def allclose(x: torch.Tensor, y: torch.Tensor, rtol=1e-3, atol=1e-3, etol=0.0):
    """allclose with error tolerance"""
    assert torch.is_tensor(x), f"x is not a tensor. {x}"
    assert torch.is_tensor(y), f"y is not a tensor. {y}"
    assert (
        x.numel() == y.numel()
    ), f"X has shape {x.shape} ({x.numel()}) but Y has shape {y.shape} ({y.numel()})"
    close = torch.allclose(x.flatten(), y.flatten(), rtol, atol)
    if not close:
        n_close = torch.isclose(x.flatten(), y.flatten(), rtol, atol).sum()
        n_el = x.numel()
        if n_close / n_el >= 1 - etol:
            return True
        print(f"Close rate {n_close}/{n_el} = {float(n_close)/n_el*100:.1f}%")
    return close
