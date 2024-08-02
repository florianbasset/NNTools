import inspect
import numbers
from functools import partial
from typing import Any, Sequence

import numpy as np


def is_image(array):
    if array.dtype == "uint8":
        if array.ndim == 2:
            return True
        if array.ndim == 3:
            if array.shape[2] in [1, 3, 4]:
                return True

    return False


def can_be_stored_as_image(array):
    if is_image(array):
        return True

    if array.dtype in [np.uint16, np.uint32, np.int16, np.int32] and array.ndim == 2:
        return array.dtype

    return False


def convert_to_image(array, original_dtype):
    h, w = array.shape[:2]
    match original_dtype:
        case np.uint16 | np.int16:
            container = np.empty((h, w, 2), dtype=np.uint8)
        case np.uint32 | np.int32:
            container = np.empty((h, w, 4), dtype=np.uint8)

    container.view(dtype=original_dtype).reshape(h, w)[:, :] = array
    container = container.reshape(h, w, -1)
    if container.shape[2] == 2:
        container = np.concatenate([container, np.zeros((h, w, 1), dtype=np.uint8)], axis=2)

    return container.reshape(h, w, -1)


def revert_image_to_original_dtype(image, original_dtype):
    h, w = image.shape[:2]
    if original_dtype in [np.uint16, np.int16]:
        image = image[:, :, :2]
    container = np.empty((h, w), dtype=original_dtype)
    container[:, :] = image.view(dtype=original_dtype).reshape(h, w)
    return container


def to_iterable(param: Any, iterable_type: Sequence[Any] = list):
    if isinstance(param, dict):
        return param
    if not isinstance(param, iterable_type):
        param = iterable_type([param])
    return param


def partial_fill_kwargs(func, list_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in list_args:
            kwargs[p.name] = list_args[p.name]
    return partial(func, **kwargs)


def call_with_filtered_kwargs(func, dict_args):
    kwargs = {}
    for p in inspect.signature(func).parameters.values():
        if p.name in dict_args:
            kwargs[p.name] = dict_args[p.name]
    return func(**kwargs)


def identity(x):
    return x


def tensor2num(x):
    if isinstance(x, numbers.Number):
        return x
    if x.dim() == 0:
        return x.item()
    else:
        return x.detach().cpu().numpy()
