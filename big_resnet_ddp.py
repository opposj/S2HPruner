import contextlib
import functools
import inspect
import itertools
import sys
import time
from itertools import zip_longest

import math
import operator
import os
import random
import re
import types
import warnings
from collections import OrderedDict, defaultdict, deque, Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import lru_cache, partial, singledispatch, update_wrapper
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    Any,
    ClassVar,
    TypeVar,
    Generic,
    Iterable,
)

import cvxpy as cp
import numpy as np
import scipy
import thop
import thop.vision.basic_hooks as th_hooks
import torch
from einops import rearrange
from xitorch.optimize import rootfinder
from xitorch import ConvergenceWarning
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from config import cfg
from stuffs import (
    broadcast_tensor,
    all_reduce_avg,
    broadcast_one_object,
    isolate_random,
)
from train import AverageMeter, accuracy, SmarterAppend

_MT = TypeVar("_MT", bound=nn.Module)


# TODO: is it suitable to add `int` into `SliceType`?
SliceType = Union[Tensor, slice]


warnings.filterwarnings("error", category=ConvergenceWarning)


# TODO: Version compatibility w.r.t. `xitorch`
from xitorch._impls.optimize.root.rootsolver import _nonlin_solver

(func_source := inspect.getsourcelines(_nonlin_solver)[0]).insert(135, "            x = xnew\n")
exec("".join(func_source), _nonlin_solver.__globals__, _nonlin_solver.__globals__)


@dataclass
class DependencyInfo:
    """
    The basic dependency information, only `out_indices` is required.
    `DependencyInfo` is used to describe runtime volatile information of a group,
    which is regarded unnecessary in the checkpoint.
    """

    out_indices: Optional[SliceType] = field(default=None)


@dataclass
class DependencyGroup(Generic[_MT]):
    """
    `DependencyGroup` is used to describe the relative more constant information
    (determined by `__getstate__`) of a group, which had better to be recorded in the checkpoint.
    Note that, `Tensor`s in `DependencyGroup` are expected to be on the same device as the `Conv2d` layers.
    """

    __states__: ClassVar[Tuple[str, ...]] = ()

    layers: Dict[str, _MT]
    dep_info: DependencyInfo | None = field(default=None)

    def __len__(self):
        return len(self.layers)

    def __getstate__(self):
        return self.state_dict()

    def __setstate__(self, state):
        self.load_state_dict(state)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "layers": {k: None for k in self.layers.keys()},
            **{k: getattr(self, k) for k in self.__states__},
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if (s_n := set(state["layers"].keys())) != (c_n := set(self.layers.keys())):
            warnings.warn(
                f"Layer names mismatched:\nLoaded: {s_n}\nCurrent: {c_n}\n"
                f"Check the checkpoint carefully before further processing.",
                RuntimeWarning,
                stacklevel=2,
            )

        del state["layers"]
        [setattr(self, k, v) for k, v in state.items()]


@dataclass(frozen=True, order=True)
class ForwardMeta(Sequence):
    """
    The basic forward meta information, only `out_indices` is required.

    Why do I use such design:
        https://devpress.csdn.net/python/62fda8a17e66823466192c66.html
    """

    out_indices: SliceType

    def __getitem__(self, i):
        return getattr(self, fields(self)[i].name)

    def __len__(self):
        return len(fields(self))


class TensorWithMeta(NamedTuple):
    """
    This class is basically for handling residual connections and `torch.flatten` in ResNet.
    """

    tensor: Tensor
    meta_data: ForwardMeta

    @classmethod
    def __torch_function__(cls, func, t_types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        dep_meta = tuple(a.meta_data for a in args if hasattr(a, "meta_data"))
        args = [a.tensor if hasattr(a, "tensor") else a for a in args]
        assert len(dep_meta) > 0
        ret = func(*args, **kwargs)
        # TODO: Handle more complex cases, where indices need to be fused.
        return TensorWithMeta(ret, dep_meta[0])

    def __str__(self):
        return "Tensor:\n{}\n\nMeta Data:\n{}".format(self.tensor, self.meta_data)

    def __add__(self, other):
        # It is assumed that the two metas are the same.
        return TensorWithMeta(self[0] + other[0], self[1])

    __iadd__ = __radd__ = __add__


class OnlyOnceContext:
    """
    This class is generally for debugging, to ensure that a certain context is only used once.
    """

    __cached = defaultdict(bool)

    def __init__(self, indicator: str):
        self.indicator = indicator

    def __enter__(self):
        assert not OnlyOnceContext.__cached[self.indicator], f"{self.indicator} is already called."

    def __exit__(self, exc_type, exc_val, exc_tb):
        OnlyOnceContext.__cached[self.indicator] = True


class MultiBindMethod:
    """
    Different from normal methods, this method has the following characteristics:
        1) When called with `(obj, Any)`: Bind the `obj` to the first argument, return a new `MultiBindMethod` object.
        2) When called with `(cls, cls)`: Bind the `cls` to the first argument, return a new `MultiBindMethod` object.
        3) When called with `(None, cls)`: Return the caller `MultibindMethod` object unchanged.
        4) Other signatures are not supported.
        5) Bindings can be applied multiple times.
        6) Support `singledispatch` according to the type of the first bound argument. (Generally the context)

    Note that:
        1) Currently, only `types.FunctionType` is supported. Any customized descriptor is not allowed.
        2) When combined with `@classmethod` or `@staticmethod`, this should be the inner decorator; however, when
        combined with `@abstractmethod`, this should be the outer decorator.
    """

    def __init__(self, func, bound=()):
        assert isinstance(func, types.FunctionType), f"Only standard function is supported, but got {type(func)}."
        self.__func__ = singledispatch(func)
        self.__bound__ = bound
        self.__update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self.__func__(*self.__bound__, *args, **kwargs)

    def __repr__(self):
        return "{module}.{cls}({func}, bound={bounds})".format(
            module=self.__class__.__module__,
            cls=self.__class__.__qualname__,
            func=self.__func__,
            bounds=self.__bound__,
        )

    def __get__(self, obj, cls=None):
        match obj, cls:
            case _, _ if isinstance(obj, cls):
                new_bound = (obj,)
            case type(), type() if obj is cls:
                new_bound = (cls,)
            case None, type():
                new_bound = ()
            case _:
                raise ValueError(f"Invalid binding pattern for <obj={obj}, cls={cls}>.")

        if new_bound:
            return self.__copy(new_bound)
        else:
            return self

    @property
    def __isabstractmethod__(self):
        return getattr(self.__func__, "__isabstractmethod__", False)

    __class_getitem__ = classmethod(types.GenericAlias)

    def register(self, cls, method=None):
        return self.__func__.register(cls, func=method)

    def __copy(self, bound=()):
        new = object.__new__(type(self))
        new.__func__ = self.__func__
        new.__update_wrapper(new, self)
        new.__bound__ = self.__bound__ + bound
        return new

    __update_wrapper = partial(update_wrapper, updated=())


class FLOPsHelper:
    @staticmethod
    def infer_output_size(
        self: nn.Conv2d,
        input: Tensor | TensorWithMeta,
        oc: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        # Infer the output size from the input size for Conv2d
        # Note that:
        #   ((output_size - 1) * stride + full_kernel_size) <= full_input_size
        #   full_input_size = input_size + 2 * padding
        #   full_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        b, _, h, w = input.size()
        assert isinstance(self.padding, tuple), "Padding must be a tuple!"
        nh = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        nw = (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        oc = self.out_channels if oc is None else oc
        return b, oc, nh, nw

    @staticmethod
    def infer_per_flops(
        self: nn.Conv2d,
        ic: Optional[int | Tensor] = None,
        oc: Optional[int | Tensor] = None,
        ks: Optional[Tuple[int, int]] = None,
        gn: Optional[int] = None,
    ) -> int | Tensor:
        ic = self.in_channels if ic is None else ic
        oc = self.out_channels if oc is None else oc
        gn = self.groups if gn is None else gn
        ks = self.kernel_size if ks is None else ks

        expand = isinstance(ic, Tensor) and isinstance(oc, Tensor)

        def _check_tensor(_x: Tensor | int, _as: Literal["input", "output"]):
            if not isinstance(_x, Tensor):
                assert isinstance(_x, int), f"Input must be a Tensor or an int, but got {_x}."
                return _x

            t_dim = _x.dim()

            if t_dim == 0:
                return _x
            elif t_dim == 1:
                if expand:
                    return _x[:, None] if _as == "input" else _x[None, :]
                else:
                    return _x
            elif t_dim == 2:
                assert (_as == "input" and _x.size(1) == 1) or (_as == "output" and _x.size(0) == 1), (
                    f"Illegal size {_x.size()} encountered."
                )
                return _x
            else:
                raise ValueError(f"Tensor must be 1D or 2D, but got {t_dim}D.")

        ic = _check_tensor(ic, "input")
        oc = _check_tensor(oc, "output")
        # Floor divide (`//`) does not support gradient, as a result, use `/` instead.
        ic_div_gn = ic / gn if isinstance(ic, Tensor) and ic.requires_grad else ic // gn
        wb_per_flops = 2 * ic_div_gn * oc * ks[0] * ks[1]

        return wb_per_flops - oc if self.bias is None else wb_per_flops

    @classmethod
    def infer_mask_flops(
        cls,
        module: nn.Conv2d,
        ic: Tensor | int,
        oc: Tensor | int,
        oh: int,
        ow: int,
        b: int,
    ) -> Tuple[Tensor | int, int]:
        # FLOPs is calculated using:
        #     1) with bias: all_flops = 2 * (i_c // g) * o_c * k_h * k_w * out_h * out_w * b
        #     2) without bias: all_flops = 2 * (i_c // g) * o_c * k_h * k_w * out_h * out_w - o_c * out_h * out_w * b
        # Note that: weight.numel() == (i_c // g) * o_c * k_h * k_w
        all_out = oh * ow * b

        masked_flops = cls.infer_per_flops(module, ic, oc) * all_out
        all_flops = cls.infer_per_flops(module) * all_out

        return masked_flops, all_flops


class GradWatcher:
    __watched: Dict[str, Dict[Literal["g_in", "g_out"], Tuple[Tensor | None]]]
    __handle: Dict[str, RemovableHandle]
    __enabled: bool
    __only_once: bool
    __accumulate: bool

    def __init__(self, enabled: bool = True, only_once: bool = True, accumulate: bool = False):
        self.__watched = {}
        self.__handle = {}
        self.__enabled = enabled
        self.__only_once = only_once
        self.__accumulate = accumulate

    @staticmethod
    def __convert(
        gi: Tuple[Tensor | None],
        go: Tuple[Tensor | None],
    ) -> Dict[Literal["g_in", "g_out"], Tuple[Tensor | None]]:
        def _(g: Tuple[Tensor | None]) -> Tuple[Tensor | None]:
            return cast(
                Tuple[Tensor | None],
                tuple(_g.detach().clone().cpu() if _g is not None else None for _g in g),
            )

        return {"g_in": _(gi), "g_out": _(go)}

    def __update(self, name, new_grad: Dict[Literal["g_in", "g_out"], Tuple[Tensor | None]]):
        def _(old_g: Tensor | None, new_g: Tensor | None):
            if old_g is None:
                return new_g
            elif new_g is None:
                return old_g
            else:
                return old_g + new_g

        if self.__accumulate:
            if name not in self.__watched:
                self.__watched.update({name: new_grad})
            else:
                new_gi = cast(
                    Tuple[Tensor | None],
                    tuple(_(old_gi, new_gi) for old_gi, new_gi in zip(self.__watched[name]["g_in"], new_grad["g_in"])),
                )
                new_go = cast(
                    Tuple[Tensor | None],
                    tuple(
                        _(old_go, new_go) for old_go, new_go in zip(self.__watched[name]["g_out"], new_grad["g_out"])
                    ),
                )
                self.__watched[name].update({"g_in": new_gi, "g_out": new_go})
        else:
            self.__watched.update({name: new_grad})

    def __watch_hook(self, name: str):
        def _(grad_inputs: Tuple[Tensor | None], grad_outputs: Tuple[Tensor | None]):
            if not self.__enabled:
                return

            self.__update(name, self.__convert(grad_inputs, grad_outputs))

            if self.__only_once:
                self.__handle[name].remove()
                del self.__handle[name]

        return _

    def watch(self, target: Tensor, name: str) -> Tensor:
        """
        This function is used to watch the gradient of a tensor.
        Note that, we use `register_hook` of the associated `Node` to watch the gradient,
        which is a bit different from the `register_hook` of a `Tensor`.

        See https://pytorch.org/docs/stable/notes/autograd.html#backward-hooks-execution for more details.
        """
        if self.__enabled:
            assert isinstance(target, Tensor), f"Target must be a Tensor, but got {type(target)}."
            self.__handle.update({name: target.grad_fn.register_hook(self.__watch_hook(name))})
        return target

    def enable(self, flag: bool = True):
        self.__enabled = flag

    def remove(self, name: str):
        assert name in self.__watched, f"Name {name} is not in the watched list."

        # For the case where `only_once` is `True`, the hook may already be removed.
        try:
            self.__handle[name].remove()
            del self.__handle[name]
        except KeyError:
            pass

        del self.__watched[name]

    def clear(self):
        for _name in self.__watched.keys():
            self.remove(_name)

    def get(self, name: str):
        return self.__watched[name]

    def get_names(self):
        return list(self.__watched.keys())

    def get_all(self):
        return self.__watched


class SliceableDeque(deque):
    """
    Copied from https://stackoverflow.com/questions/7064289/use-slice-notation-with-collections-deque.
    """

    def __getitem__(self, s):
        try:
            start, stop, step = s.start or 0, s.stop or sys.maxsize, s.step or 1
        except AttributeError:  # not a slice but an int
            return super().__getitem__(s)
        try:  # normal slicing
            return list(itertools.islice(self, start, stop, step))
        except ValueError:  # incase of a negative slice object
            length = len(self)
            start, stop = (
                length + start if start < 0 else start,
                length + stop if stop < 0 else stop,
            )
            return list(itertools.islice(self, start, stop, step))


def make_divisible(v, divisor=8):
    return int((v + divisor - 1) // divisor * divisor)


def make_divisible_v2(v: Tensor, limit: Tensor) -> Tensor:
    limit_cap = max(1, int(limit * cfg.msk_config.ip_enforce_ch_frac))
    divisor = cfg.msk_config.ip_enforce_ch
    while divisor > limit_cap:
        divisor = max(1, divisor // 2)
    if GB_MAKE_DIVISIBLE_TENSOR_DIVISOR is not None:
        divisor = GB_MAKE_DIVISIBLE_TENSOR_DIVISOR
    # noinspection PyTypeChecker
    return min(limit, (v + divisor - 1) // divisor * divisor)


GB_MAKE_DIVISIBLE_TENSOR_DIVISOR = None


def make_divisible_tensor(v: Tensor, divisor=8, *, limit: int | Tensor | None = None) -> Tensor:
    # noinspection PyTypeChecker
    # TODO: Better control for small channels.
    if cfg.model == "vit_9" and v <= 16:
        divisor = 1

    if GB_MAKE_DIVISIBLE_TENSOR_DIVISOR is not None:
        divisor = GB_MAKE_DIVISIBLE_TENSOR_DIVISOR

    if limit is not None:
        return min(limit, (v + divisor - 1) // divisor * divisor)
    else:
        # noinspection PyTypeChecker
        return (v + divisor - 1) // divisor * divisor


def recumsum(x: Tensor, dim: int = 0, fast: bool = False) -> Tensor:
    if fast:
        temp = torch.cumsum(x, dim=dim)
        return x - temp + temp[-1:None]

    return torch.cumsum(x.flip(dims=(dim,)), dim=dim).flip(dims=(dim,))


def exec_range(
    fn_range: Callable[[int], bool],
    is_epoch: bool = True,
    is_iter: bool = False,
    ep_only_once: bool = False,
    only_once: bool = False,
):
    assert is_epoch is False or is_iter is False, "Epoch range and iteration range cannot be set at the same time."
    assert ep_only_once is False or only_once is False, (
        "Total `only_once` and epoch `ep_only_once` cannot be set at the same time."
    )

    def _decorator(f_n):
        if is_epoch:
            if ep_only_once:
                ep_exec_cache = set()

                def _wrapper(*args, **kwargs):
                    rt_epoch = cfg.rt_epoch
                    if fn_range(rt_epoch) and rt_epoch not in ep_exec_cache:
                        ep_exec_cache.add(rt_epoch)
                        return f_n(*args, **kwargs)

            elif only_once:
                executed = False

                def _wrapper(*args, **kwargs):
                    nonlocal executed
                    if not executed and fn_range(cfg.rt_epoch):
                        executed = True
                        return f_n(*args, **kwargs)

            else:

                def _wrapper(*args, **kwargs):
                    if fn_range(cfg.rt_epoch):
                        return f_n(*args, **kwargs)

        else:
            if only_once:
                executed = False

                def _wrapper(*args, **kwargs):
                    nonlocal executed
                    if not executed and fn_range(cfg.rt_iter):
                        executed = True
                        return f_n(*args, **kwargs)

            else:

                def _wrapper(*args, **kwargs):
                    if fn_range(cfg.rt_iter):
                        return f_n(*args, **kwargs)

        update_wrapper(_wrapper, f_n)
        return _wrapper

    return _decorator


# TODO: Improve the matching logic
def get_supers():
    """
    This function is used to select the appropriate mixins for the current configuration.
    Bi-matching is used to select the appropriate mixins, i.e., for each configuration, the corresponding mixin is
    the one that matches most of the configuration items.
    """
    cfgs = dict(vars(cfg))

    def _match(_mixin):
        _match_num = 0
        for _cn, _cv in cfgs.items():
            if hasattr(_mixin, f"__{_cn}__"):
                if getattr(_mixin, f"__{_cn}__") == _cv:
                    _match_num += 1
                else:
                    _match_num = 0
                    break
        return _match_num

    mixins = dict(
        filter(
            lambda _var: isinstance(_var[1], type) and issubclass(_var[1], MixIn) and _match(_var[1]) > 0,
            globals().items(),
        )
    )

    matched_pairs = {
        _cn: _tmp
        for _cn, _cv in cfgs.items()
        if (
            _tmp := list(
                filter(
                    lambda _mixin: hasattr(_mixin, f"__{_cn}__") and getattr(_mixin, f"__{_cn}__") == _cv,
                    mixins.values(),
                )
            )
        )
    }
    inv_matched_pairs = {
        _mixin: _tmp
        for _mn, _mixin in mixins.items()
        if (
            _tmp := list(
                filter(
                    lambda _cn: hasattr(_mixin, f"__{_cn}__") and getattr(_mixin, f"__{_cn}__") == getattr(cfg, _cn),
                    cfgs.keys(),
                )
            )
        )
    }

    filtered_matched_pairs = {
        _cn: _tmp[0] if len(_tmp) == 1 else _tmp[int(np.argsort([len(inv_matched_pairs[_t]) for _t in _tmp])[-1])]
        for _cn, _tmp in matched_pairs.items()
    }
    selected_mixins = set(filtered_matched_pairs.values())

    cfg.rt_logger.info(
        f"Mixin selection:\n"
        + f"{'-' * 100}\n"
        + "\n".join(
            [
                f"{_mixin} -> {', '.join([f'{_cn}={cfgs[_cn]}' for _cn in inv_matched_pairs[_mixin]])}"
                for _mixin in selected_mixins
            ]
        )
        + f"\n{'-' * 100}"
    )

    return tuple(selected_mixins) + (DynamicResNetBase,)


def create_dynamic_resnet(**kwargs):
    class DynamicResNet(*get_supers()):
        def _post_initialize(self):
            for _super in self.__class__.__bases__:
                if (su_init := getattr(_super, "_post_initialize", None)) is not None:
                    su_init(self)

    # Sanity check, ensure that no attribute is overlapped in different mixins.
    public_attrs = dir(MixIn)
    bases_attrs = [dir(_base) for _base in DynamicResNet.__bases__][:-1]
    overlapped_attrs = []
    for _attr in dir(DynamicResNet):
        if _attr not in public_attrs and sum([_attr in _base_attrs for _base_attrs in bases_attrs]) > 1:
            overlapped_attrs.append(_attr)
    if overlapped_attrs:
        raise ValueError(f"Attribute(s) {overlapped_attrs} overlapped in different mixins.")

    return DynamicResNet


class DynamicResNetBase(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        # Save the arguments for `copy` and `deepcopy`,
        # as we need to initialize the model with the same arguments.
        self._saved_args = locals()

        super().__init__(
            block,
            layers,
            num_classes,
            zero_init_residual,
            groups,
            width_per_group,
            replace_stride_with_dilation,
            norm_layer,
        )

        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=True)
        else:
            self.dropout = None

        # Note that `forward_mode` is set automatically via `set_running_subnet` and `reset_running_subnet` by default.
        self.forward_mode: Literal["teacher", "student"] = "teacher"
        self.stage_list = layers
        self.proxy = None

        # Convert to sync batch norm. By the way, we do not need to track the running stats.
        nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.apply(lambda _m: setattr(_m, "track_running_stats", False) if isinstance(_m, nn.SyncBatchNorm) else None)

        # This can be used to initialize the proxy and convert the model to dynamic, etc.
        self._post_initialize()

    def copy(self):
        device = next(self.parameters()).device
        new_self = self.__class__(**self._saved_args).to(device)
        new_self.__dict__.update(self.__dict__)  # Is it robust to do this?
        new_self.load_state_dict(self.state_dict())
        return new_self

    def forward(self, x: Tensor) -> Tensor:
        if self.forward_mode == "teacher":
            return self._forward_impl_teacher(x)
        elif self.forward_mode == "student":
            return self._forward_impl_student(x)
        else:
            raise ValueError("forward mode must be teacher or student")

    def set_running_subnet(self, proxy):
        self.proxy = proxy
        self.forward_mode = "student"
        self._apply_proxy(proxy)

    def reset_running_subnet(self):
        self._clear_proxy()
        self.forward_mode = "teacher"
        self.proxy = None

    def get_all_subnets(self, **kwargs):
        raise NotImplementedError

    def get_test_subnets(self, **kwargs):
        yield from self.get_all_subnets()

    def sample_subnet(self, **kwargs):
        all_subnets = self.get_all_subnets(**kwargs)
        return all_subnets[np.random.randint(0, len(all_subnets))]

    def optim_setup_hook(self, optimizer, scheduler):
        return optimizer, scheduler

    def post_load_checkpoint(self, checkpoint):
        return checkpoint

    def pre_save_checkpoint(self, checkpoint):
        return checkpoint

    @staticmethod
    def student_only(t_func=None, s_context=True, t_context=False):
        assert t_func is None or isinstance(t_func, types.FunctionType), "`t_func` must be `FunctionType` or `None`."

        if t_func is None:

            def t_func(*args, **kwargs):
                pass

        match s_context, t_context:
            case True, False:

                def _adapt_args(mode, args):
                    return args if mode == "student" else args[1:]

            case False, True:

                def _adapt_args(mode, args):
                    return args if mode != "student" else args[1:]

            case True, True:

                def _adapt_args(mode, args):
                    return args

            case False, False:

                def _adapt_args(mode, args):
                    return args[1:]

            case _, _:
                raise ValueError(f"Illegal `s_context` and `t_context` as {s_context} and {t_context}.")

        def _decorator(s_func):
            assert isinstance(s_func, types.FunctionType), "`s_func` must be `FunctionType`."

            @functools.wraps(s_func)
            def _wrapper(*args, **kwargs):
                assert isinstance(ctx := cast(DynamicResNetBase, args[0]), DynamicResNetBase), (
                    "The first argument must be of `DynamicResNetBase` type."
                )

                mode = ctx.forward_mode
                args = _adapt_args(mode, args)

                if mode == "student":
                    return s_func(*args, **kwargs)
                else:
                    return t_func(*args, **kwargs)

            return _wrapper

        return _decorator

    def _post_initialize(self):
        pass

    def _apply_proxy(self, proxy):
        pass

    def _clear_proxy(self):
        pass

    def _forward_impl_teacher(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x

    _forward_impl_student = _forward_impl_teacher


class MixIn:
    __DynamicResNet: str = "Union[MixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        pass


class ProxyMixIn(MixIn):
    """
    Hook execution order: `optim_setup_hook` -> `post_load_checkpoint` -> `pre_save_checkpoint`.
    """

    __dimension__: str
    __DynamicResNet = "Union[ProxyMixIn, DynamicResNetBase]"

    def _apply_proxy(self: __DynamicResNet, proxy: Any):
        pass

    def _clear_proxy(self: __DynamicResNet):
        pass

    def optim_setup_hook(
        self: __DynamicResNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        return optimizer, scheduler

    def post_load_checkpoint(self: __DynamicResNet, checkpoint: dict) -> dict:
        """
        Return a modified checkpoint for further usage or setting extra stuffs for `self`.
        """
        return checkpoint

    def pre_save_checkpoint(self: __DynamicResNet, checkpoint: dict) -> dict:
        """
        Return a modified checkpoint for saving
        """
        return checkpoint


class SpaceMixIn(MixIn):
    __proxy_space__: str
    __DynamicResNet = "Union[SpaceMixIn, DynamicResNetBase]"

    def get_all_subnets(self: __DynamicResNet, **kwargs):
        return DynamicResNetBase.get_all_subnets(self, **kwargs)

    def get_test_subnets(self: __DynamicResNet, **kwargs):
        return DynamicResNetBase.get_test_subnets(self, **kwargs)

    def sample_subnet(self: __DynamicResNet, **kwargs):
        return DynamicResNetBase.sample_subnet(self, **kwargs)


class BasicDepthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "d_basic"
    __DynamicResNet = "Union[BasicDepthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        return list(itertools.product(*[range(1, _i + 1) for _i in self.stage_list]))


class BasicDepthProxyMixIn(ProxyMixIn):
    __dimension__ = "d_basic"
    __DynamicResNet = "Union[BasicDepthProxyMixIn, DynamicResNetBase]"

    class Sequential(nn.Sequential):
        _depth: int

        def _get_out_indices(
            self,
            input: Tensor | TensorWithMeta,
            depth: int,
        ) -> SliceType: ...

    def _post_initialize(self: __DynamicResNet):
        # Postponed class modification.
        self.Sequential.__getitem__ = self._sequential_get_item
        self.Sequential.forward = self._sequential_forward
        self.Sequential._get_out_indices = self._sequential_get_out_indices

        for l_name, layer in zip(
            ("layer1", "layer2", "layer3", "layer4"),
            (self.layer1, self.layer2, self.layer3, self.layer4),
        ):
            layer.__class__ = self.Sequential

    def _apply_proxy(self: __DynamicResNet, proxy):
        for i, layer in enumerate((self.layer1, self.layer2, self.layer3, self.layer4)):
            cast(self.Sequential, layer)._depth = proxy[i]

    def _clear_proxy(self: __DynamicResNet):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            layer = cast(self.Sequential, layer)
            if hasattr(layer, "_depth"):
                del layer._depth

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.Sequential.__getitem__)
    def _sequential_get_item(
        ctx: __DynamicResNet,
        self: Sequential,
        idx: SliceType,
    ):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        elif isinstance(idx, torch.Tensor):
            # Sort the indices to make sure we get them in order
            new_modules = operator.itemgetter(*idx.sort()[0])(list(self._modules.items()))
            new_modules = [new_modules] if isinstance(new_modules[1], nn.Module) else new_modules
            return self.__class__(OrderedDict(new_modules))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.Sequential.forward)
    def _sequential_forward(
        ctx: __DynamicResNet,
        self: Sequential,
        input,
    ):
        indices = self._get_out_indices(input, self._depth)
        for module in self[indices]:
            input = module(input)
        return input

    @MultiBindMethod
    def _sequential_get_out_indices(
        ctx: __DynamicResNet,
        self: Sequential,
        input: Tensor | TensorWithMeta,
        depth: int,
    ) -> SliceType:
        return slice(None, depth)


class BasicWidthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "w_basic"
    __DynamicResNet = "Union[BasicWidthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        return list(itertools.product([0.25, 0.5, 0.75, 1], repeat=len(self.stage_list)))


class ManualWidthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "w_manual"
    __DynamicResNet = "Union[ManualWidthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        repeat_num = len(self.stage_list)
        return list(itertools.product(np.linspace(0.3, 1, repeat_num), repeat=repeat_num))


class BasicWidthProxyMixIn(ProxyMixIn):
    __dimension__ = "w_basic"
    __DynamicResNet = "Union[BasicWidthProxyMixIn, DynamicResNetBase]"

    @dataclass
    class DependencyInfo(DependencyInfo):
        width: float = field(default=1.0)

    class DependencyGroup(DependencyGroup[nn.Conv2d]): ...

    ForwardMeta = ForwardMeta

    class Linear(nn.Linear):
        def _core_forward(
            self,
            input: Tensor,
            in_indices: Optional[SliceType] = None,
            out_indices: Optional[SliceType] = None,
        ) -> Tensor: ...

    class SyncBatchNorm(nn.SyncBatchNorm):
        def _core_forward(
            self,
            input: Tensor,
            in_indices: Optional[SliceType] = None,
        ) -> Tensor: ...

    class Conv2d(nn.Conv2d):
        _group: DependencyGroup

        def _core_forward(
            self,
            input: Tensor,
            in_indices: Optional[SliceType] = None,
            out_indices: Optional[SliceType] = None,
        ) -> Tensor: ...

        def _get_oc_rep(
            self,
            input: Tensor,
            in_meta: ForwardMeta,
            dep_info: DependencyInfo,
        ) -> int | Tensor:
            """
            Return a representative for output channels in the form of either an `int` or a `Tensor`.
            """
            ...

        def _get_out_indices(
            self,
            input: Tensor,
            in_meta: ForwardMeta,
            dep_info: DependencyInfo,
            new_out: int | Tensor,
        ) -> SliceType:
            """
            Return a representative for output indices in the form of either the order of output channels or a `slice`.
            """
            ...

    def _post_initialize(self: __DynamicResNet):
        self._dep_groups = [self._set_group(_group) for _group in self._dependency_extract()]

        # Postponed class modification.
        self.Linear._core_forward = self._fc_core_forward
        self.Linear.forward = self._fc_forward
        self.SyncBatchNorm._core_forward = self._sync_bn_core_forward
        self.SyncBatchNorm.forward = self._sync_bn_forward
        self.Conv2d._core_forward = self._conv2d_core_forward
        self.Conv2d.forward = self._conv2d_forward
        self.Conv2d._get_oc_rep = self._conv2d_get_oc_rep
        self.Conv2d._get_out_indices = self._conv2d_get_out_indices

        # Register forward methods to Conv2d, SyncBatchNorm and the last linear layer.
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and n != "conv1":
                m.__class__ = self.Conv2d
                # Hence the `prepend` is used here,
                # it is expected that all following registered pre-hooks not using `prepend`.
                try:
                    m.register_forward_pre_hook(cast(Callable, self.__conv2d_forward_pre_hook), prepend=True)
                except TypeError:
                    # For compatibility with PyTorch 1.x
                    m.register_forward_pre_hook(cast(Callable, self.__conv2d_forward_pre_hook))

            elif isinstance(m, nn.SyncBatchNorm) and n != "bn1":
                m.__class__ = self.SyncBatchNorm
            elif n == "fc":
                m.__class__ = self.Linear

        # Register some `torch.nn.functional` functions
        self._register_indexed_torch_functional("relu")
        self._register_indexed_torch_functional("adaptive_avg_pool2d")

        # Alter some `thop` hooks
        from .mobilenet_v3_ddp import FLOPsHelper as LinearFLOPsHelper

        def _count_linear(module: nn.Linear, input, output):
            input = input[0]
            if isinstance(input, TensorWithMeta):
                input = input.tensor
            if isinstance(output, TensorWithMeta):
                output = output.tensor

            module.total_ops += LinearFLOPsHelper.infer_mask_flops(
                module,
                input.size(-1),
                output.size(-1),
                *(1, 1),
                functools.reduce(operator.mul, input.size()[:-1]),
            )[0]

        def _count_conv2d(module: nn.Conv2d, input, output):
            input = input[0]
            if isinstance(input, TensorWithMeta):
                input = input.tensor
            if isinstance(output, TensorWithMeta):
                output = output.tensor

            module.total_ops += FLOPsHelper.infer_mask_flops(
                module,
                input.size(1),
                output.size(1),
                output.size(2),
                output.size(3),
                output.size(0),
            )[0]

        thop.profile = partial(
            thop.profile,
            custom_ops={
                nn.Conv2d: self._get_indexed_thop_hook(th_hooks, "zero_ops"),
                nn.SyncBatchNorm: self._get_indexed_thop_hook(th_hooks, "zero_ops"),
                self.Conv2d: _count_conv2d,
                self.SyncBatchNorm: self._get_indexed_thop_hook(th_hooks, "count_normalization"),
                nn.AdaptiveAvgPool2d: self._get_indexed_thop_hook(th_hooks, "count_adap_avgpool"),
                self.Linear: _count_linear,
            },
        )

    def _apply_proxy(self: __DynamicResNet, proxy):
        """
        Layer-wise width proxy.
        """
        for _dep_group in self._dep_groups:
            layer_idx = set(
                map(
                    lambda nm: int(re.search(r"layer(?P<index>\d)", nm[0])["index"]),
                    _dep_group.layers.items(),
                )
            )
            assert len(layer_idx) == 1, "Layer index should be unique in a dependency group."
            layer_idx = layer_idx.pop()
            _dep_group.dep_info = self.DependencyInfo(width=proxy[layer_idx - 1])

    def _clear_proxy(self: __DynamicResNet):
        for _dep_group in self._dep_groups:
            _dep_group.dep_info = None

    def _dependency_extract(self: __DynamicResNet) -> List[Dict[str, nn.Conv2d]]:
        # TODO: requires more robust implementation
        dep_groups = []
        latest_group = None
        latest_channel_num = None
        for n, m in self.named_modules():
            if isinstance(m, BasicBlock) or isinstance(m, Bottleneck):
                free_layers = filter(lambda nx_x: isinstance(nx_x[1], nn.Conv2d), m.named_children())
                # Maximum free width for each layer (except downsample layer) in the block
                if m.downsample is not None:
                    for _n, _m in free_layers:
                        dep_groups.append({f"{n}.{_n}": _m})
                        latest_channel_num = _m.out_channels
                    latest_group = dep_groups[-1]
                    # The downsample layer should share the output of the last conv layer
                    assert isinstance(m.downsample[0], nn.Conv2d), "Weird downsample layer encountered."
                    assert latest_channel_num == m.downsample[0].out_channels, (
                        f"Downsample layer should share the output of the last conv layer in block {n}."
                        f"Got {latest_channel_num} and {m.downsample[0].out_channels} instead."
                    )
                    latest_group.update({f"{n}.downsample.0": m.downsample[0]})
                # Sub-maximum free width without the last conv layer in the block
                else:
                    cur_name, cur_layer = next(free_layers)
                    while (next_name_layer := next(free_layers, None)) is not None:
                        assert isinstance(cur_layer, nn.Conv2d) and isinstance(next_name_layer[1], nn.Conv2d), (
                            "Only conv layer is considered in dependency extraction。"
                        )
                        dep_groups.append({f"{n}.{cur_name}": cur_layer})
                        cur_name, cur_layer = next_name_layer
                    # The last conv layer should share the depth of the previous last conv layer
                    assert cur_layer.out_channels == latest_channel_num, (
                        f"Erroneous dependency group detected in block {n}."
                    )
                    latest_group.update({f"{n}.{cur_name}": cur_layer})

        return dep_groups

    def _set_group(self: __DynamicResNet, group: Dict[str, nn.Conv2d]):
        gp = self.DependencyGroup(group)
        for _name, _layer in group.items():
            cast(self.Conv2d, _layer)._group = gp
        return gp

    @staticmethod
    def _register_indexed_torch_functional(func_name):
        # Always assume that the input is the first argument
        # Implicitly realize student-teacher dispatch due to the divergence on `TensorWithMeta`
        _ori_func = getattr(F, func_name)

        def _new_func(input: TensorWithMeta | Tensor, *args, **kwargs):
            if isinstance(input, TensorWithMeta):
                input, in_dep = input
                return TensorWithMeta(_ori_func(input, *args, **kwargs), in_dep)
            else:
                return _ori_func(input, *args, **kwargs)

        setattr(F, func_name, _new_func)

    @staticmethod
    def _get_indexed_thop_hook(hook_module, hook_name):
        # Always assume that the input is the first argument
        # Implicitly realize student-teacher dispatch due to the divergence on `TensorWithMeta`
        _ori_hook = getattr(hook_module, hook_name)

        def _new_hook(module: nn.Module, args, output):
            input = args[0]
            if isinstance(input, TensorWithMeta):
                input, output = input[0], output[0]
                args = (input,) + args[1:]

            _ori_hook(module, args, output)

        return _new_hook

    @DynamicResNetBase.student_only()
    def __conv2d_forward_pre_hook(
        ctx: __DynamicResNet,
        module: Conv2d,
        args: Tuple[Tensor | TensorWithMeta],
    ):
        # This should be called only once for initialization
        input = args[0]
        if isinstance(input, Tensor):
            input = TensorWithMeta(
                input,
                ctx.ForwardMeta(torch.arange(input.shape[1], device=input.device)),
            )
        return (input,)

    @MultiBindMethod
    def _fc_core_forward(
        ctx: __DynamicResNet,
        self: Linear,
        input: Tensor,
        in_indices: SliceType = None,
        out_indices: SliceType = None,
    ) -> Tensor:
        weight, bias = self.weight, self.bias
        if out_indices is not None:
            weight = weight[out_indices]
            bias = bias[out_indices] if bias is not None else None
        if in_indices is not None:
            weight = weight[:, in_indices]

        return F.linear(input, weight, bias)

    _req_param_stat: bool
    _param_stat: list[float]

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.Linear.forward)
    def _fc_forward(
        ctx: __DynamicResNet,
        self: Linear,
        input: TensorWithMeta,
    ) -> Tensor:
        # It is assumed that there is only one (last) Linear layer in the model.
        input, in_dep = input
        output = self._core_forward(input, in_dep.out_indices)

        # TODO: What if `_req_param_stat` is not defined?
        if ctx._req_param_stat:
            total_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
            real_params = input.size(-1) * output.size(-1) + (output.size(-1) if self.bias is not None else 0)
            ctx._param_stat[0] += real_params
            ctx._param_stat[1] += total_params

        return output

    @MultiBindMethod
    def _sync_bn_core_forward(
        ctx: __DynamicResNet,
        self: SyncBatchNorm,
        input: Tensor,
        in_indices: SliceType = None,
    ) -> Tensor:
        process_group = dist.group.WORLD
        if self.process_group:
            process_group = self.process_group
        world_size = dist.get_world_size(process_group)

        weight, bias = self.weight, self.bias
        # Buffered running/var to ensure the update of self.running_mean/var
        buf_running_mean = self.running_mean if not self.training or self.track_running_stats else None
        buf_running_var = self.running_var if not self.training or self.track_running_stats else None

        if in_indices is not None:
            weight, bias = weight[in_indices], bias[in_indices]
            if buf_running_mean is not None:
                buf_running_mean = buf_running_mean[in_indices]
            if buf_running_var is not None:
                buf_running_var = buf_running_var[in_indices]

        if not self.training or world_size == 1:
            result = F.batch_norm(
                input,
                buf_running_mean,
                buf_running_var,
                weight,
                bias,
                self.training,
                self.momentum,
                self.eps,
            )

        else:
            result = sync_batch_norm.apply(
                input,
                weight,
                bias,
                buf_running_mean,
                buf_running_var,
                self.eps,
                self.momentum,
                process_group,
                world_size,
            )

        # Update the running mean/var based on the buffered ones
        if in_indices is not None and self.track_running_stats and self.training:
            self.running_mean.scatter_(0, in_indices, buf_running_mean)
            self.running_var.scatter_(0, in_indices, buf_running_var)

        return result

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.SyncBatchNorm.forward)
    def _sync_bn_forward(
        ctx: __DynamicResNet,
        self: SyncBatchNorm,
        input: TensorWithMeta,
    ) -> TensorWithMeta:
        input, in_dep = input
        output = self._core_forward(input, in_dep.out_indices)

        # TODO: What if `_req_param_stat` is not defined?
        if ctx._req_param_stat:
            total_params, real_params = 0, 0

            if self.weight is not None:
                total_params += self.weight.numel()
                real_params += output.size(1)
            if self.bias is not None:
                total_params += self.bias.numel()
                real_params += output.size(1)
            if self.running_mean is not None:
                total_params += self.running_mean.numel()
                real_params += output.size(1)
            if self.running_var is not None:
                total_params += self.running_var.numel()
                real_params += output.size(1)

            ctx._param_stat[0] += real_params
            ctx._param_stat[1] += total_params

        return TensorWithMeta(output, in_dep)

    @MultiBindMethod
    def _conv2d_core_forward(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_indices: SliceType = None,
        out_indices: SliceType = None,
    ) -> Tensor:
        weight, bias = self.weight, self.bias
        if out_indices is not None:
            weight = weight[out_indices]
            bias = bias[out_indices] if bias is not None else None
        # TODO: Better handling group-wise convolution.
        if in_indices is not None and self.groups == 1:
            weight = weight[:, in_indices]

        if self.groups == 1:
            return self._conv_forward(input, weight, bias)
        else:
            prev_gp, self.groups = self.groups, weight.size(0)
            out = self._conv_forward(input, weight, bias)
            self.groups = prev_gp
            return out

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.Conv2d.forward)
    def _conv2d_forward(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: TensorWithMeta,
    ) -> TensorWithMeta:
        in_meta: ctx.ForwardMeta
        input, in_meta = input.tensor, input.meta_data
        dep_info: Optional[ctx.DependencyInfo] = self._group.dep_info

        if dep_info.out_indices is None:
            new_out = self._get_oc_rep(input, in_meta, dep_info)
            dep_info.out_indices = self._get_out_indices(input, in_meta, dep_info, new_out)

        in_indices = in_meta.out_indices
        out_indices = dep_info.out_indices

        output = self._core_forward(input, in_indices, out_indices)

        if ctx._req_param_stat:
            total_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
            real_params = input.size(1) * output.size(1) * self.kernel_size[0] * self.kernel_size[1] + (
                output.size(1) if self.bias is not None else 0
            )
            if self.groups != 1:
                real_params /= output.size(1)
            ctx._param_stat[0] += real_params
            ctx._param_stat[1] += total_params

        return TensorWithMeta(output, ctx.ForwardMeta(out_indices))

    @MultiBindMethod
    def _conv2d_get_oc_rep(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        dep_info: DependencyInfo,
    ) -> int:
        return make_divisible(dep_info.width * self.out_channels)

    @MultiBindMethod
    def _conv2d_get_out_indices(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        dep_info: DependencyInfo,
        new_out: int,
    ) -> SliceType:
        return torch.arange(new_out, device=input.device)


class MaskedWidthProxySpaceMixIn(BasicWidthProxyMixIn, SpaceMixIn):
    __dimension__ = "w_masked"
    __proxy_space__ = "w_masked"
    __DynamicResNet = "Union[MaskedWidthProxySpaceMixIn, DynamicResNetBase]"

    __req_hard_forward: bool
    __req_soft_forward: bool

    _all_flops: float = 0.0
    _all_soft_flops: Tensor | float = 0.0
    _all_norm: Tensor | float = 0.0
    _all_entropy: Tensor | float = 0.0

    _flops_losses = AverageMeter(epoch_fresh=True)
    _norm_losses = AverageMeter(epoch_fresh=True)
    _entropy_losses = AverageMeter(epoch_fresh=True)
    _aux_losses = AverageMeter(epoch_fresh=True)
    _timer = defaultdict(lambda: AverageMeter(epoch_fresh=True))

    _gw = GradWatcher(enabled=cfg.msk_config.enable_gw)

    # TODO: Better alignment for `__max` and `self`
    class Proxy(dict[str, int]):
        _cur: dict[int, int]
        __max: Tuple[int, ...]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cur = {}

        def __repr__(self):
            return f"Proxy({super().__repr__()})"

        def __str__(self):
            # For logging, to save space and be more concise, we only print the values/max_values.
            return (
                "<"
                + " ".join(
                    f"{_cur:>{_dit}}/{_max:<{_dit}}"
                    for (_cur, _max, _dit) in itertools.starmap(
                        lambda _k, _c: (_c, self.__max[_k], len(str(self.__max[_k]))),
                        self._cur.items(),
                    )
                )
                + ">"
            )

        def extract(self) -> Tuple[int, ...]:
            return tuple(self.values())

        def load(self, proxy: Tuple[int, ...]) -> "MaskedWidthProxySpaceMixIn.Proxy":
            [self.__setitem__(_name, _cur) for _name, _cur in zip(self.keys(), proxy)]
            return self

        def grow(self) -> "MaskedWidthProxySpaceMixIn.Proxy":
            return deepcopy(self).load(
                tuple(
                    make_divisible(
                        random.choice(range(min(cur + 1, self.__max[key]), self.__max[key] + 1)),
                        cfg.msk_config.ch_divisor,
                    )
                    for key, cur in self._cur.items()
                )
            )

        @classmethod
        def set_max(cls, c_max: Tuple[int, ...]):
            cls.__max = c_max

    class SubnetInfo:
        idx: int | None
        proxy: "MaskedWidthProxySpaceMixIn.Proxy"
        dis_loss: AverageMeter
        cal_num: int
        mut_num: int
        mut_scale: float
        mut_history: deque[Tuple[int, ...]]
        __last_iter: int
        __dep_groups: List["MaskedWidthProxySpaceMixIn.DependencyGroup"]

        def __init__(
            self,
            idx: int | None,
            proxy: "MaskedWidthProxySpaceMixIn.Proxy",
            history_num: int | None = 0,
        ):
            self.idx = idx
            self.proxy = proxy
            self.dis_loss = AverageMeter(avg_window=len(cfg.rt_train_loader))
            self.cal_num = 0
            self.mut_num = 0
            self.mut_scale = cfg.msk_config.param_grad_scale
            self.mut_history = deque(maxlen=history_num)
            self.__last_iter = cfg.rt_iter

        @classmethod
        def set_groups(cls, groups: List["MaskedWidthProxySpaceMixIn.DependencyGroup"]):
            cls.__dep_groups = groups

        def copy_masks(self):
            """
            Return the copy of all masks with `self.idx` in the dependency groups.
            """
            return [gp.mask[self.idx].detach().clone() for gp in self.__dep_groups]

        def named_masks(self) -> Iterable[Tuple[str, nn.Parameter]]:
            return ((next(iter(gp.layers.keys())), gp.mask[self.idx]) for gp in self.__dep_groups)

        @property
        def masks(self):
            """
            Return the reference of all masks with `self.idx` in the dependency groups.
            """
            return [gp.mask[self.idx] for gp in self.__dep_groups]

        @masks.setter
        def masks(self, masks: List[nn.Parameter]):
            """
            Inplace set the masks with `self.idx` in the dependency groups.
            """
            [gp.mask[self.idx].copy_(mask) for gp, mask in zip(self.__dep_groups, masks)]

        if cfg.msk_config.solidify:

            @property
            def non_solidified_masks(self):
                """
                Return the reference of all non-solidified masks with `self.idx` in the dependency groups.
                """
                return [gp.mask[self.idx] for gp in self.__dep_groups if not gp.solidified[self.idx]]

        def update_loss(self, loss: Tensor | float, count: int = 1):
            # Although the `AverageMeter` can handle both the `Tensor` and `float` type, we still
            # explicitly convert the `Tensor` type to `float` for convenience.
            if isinstance(loss, Tensor):
                loss = loss.item()

            self.dis_loss.update(loss, count)

        def backup(
            self,
        ) -> Tuple["MaskedWidthProxySpaceMixIn.Proxy", List[nn.Parameter]]:
            return deepcopy(self.proxy), self.copy_masks()

        def restore(self, backup: Tuple["MaskedWidthProxySpaceMixIn.Proxy", List[nn.Parameter]]):
            self.proxy, self.masks = backup

            if cfg.msk_config.dy_pgs and (cfg.rt_iter - self.__last_iter) > cfg.msk_config.pgs_interval:
                self.__last_iter = cfg.rt_iter
                self.mut_scale = min(
                    self.mut_scale * cfg.msk_config.pgs_multiplier,
                    cfg.msk_config.pgs_max,
                )

        def mutate(self, backup: Tuple["MaskedWidthProxySpaceMixIn.Proxy", List[nn.Parameter]]):
            self.mut_num += 1
            self.__last_iter = cfg.rt_iter
            self.mut_scale = cfg.msk_config.param_grad_scale
            self.mut_history.append(backup[0].extract())

        @torch.no_grad()
        def extract(self) -> Tuple[int, ...]:
            def _(_gp: MaskedWidthProxySpaceMixIn.DependencyGroup):
                norm_mask = F.softmax(torch.div(_gp.mask[self.idx], cfg.msk_config.softmax_temp))
                soft_out_channels = MaskedWidthProxySpaceMixIn.STECeil.apply(
                    torch.dot(norm_mask, _gp.channel_nums.float())
                )
                return int(soft_out_channels.item())

            return tuple(map(_, self.__dep_groups))

        def __repr__(self):
            return repr(self.proxy)

        def __str__(self):
            return str(self.proxy)

        def __eq__(self, other):
            return self.proxy == other.proxy

    _cur_subnet: SubnetInfo
    _pool: List[SubnetInfo] = []

    @dataclass
    class DependencyInfo(DependencyInfo):
        soft_out_channels: Optional[Tensor] = field(default=None)
        diff_mask: Optional[Tensor] = field(default=None)

    @dataclass
    class DependencyGroup(BasicWidthProxyMixIn.DependencyGroup):
        # TODO: `channel_nums` and `ch_divisor` should not be loaded when resumed from a pretrained model.
        __states__ = (
            "l1_order",
            "order_initialized",
            "mask",
            "gp_flops",
            "channel_nums",
            "ch_divisor",
        )
        __optimizer: ClassVar[Optional[torch.optim.Optimizer]] = None
        __auto_add: ClassVar[bool] = False

        channel_nums: Optional[Tensor] = field(default=None)
        l1_order: Optional[Tensor] = field(default=None)
        order_initialized: Optional[Tensor] = field(default=None)
        mask: Optional[Dict[int, nn.Parameter]] = field(default=None)
        ch_divisor: Optional[Tensor] = field(default=None)
        gp_flops: Dict[any, Tuple[float, float]] = field(default_factory=dict)

        # Solidification-specific attribute
        if cfg.msk_config.solidify:
            __states__ += ("solidified",)
            solidified: defaultdict[int, bool] = field(default_factory=lambda: defaultdict(bool))

        if cfg.msk_config.iterative_pruning:
            __states__ += ("prune_metric", "sub_flops")
            if cfg.msk_config.ip_dw:
                prune_metric: defaultdict[str, defaultdict[tuple[int, ...], AverageMeter]] = field(
                    default_factory=lambda: defaultdict(
                        defaultdict(
                            partial(
                                AverageMeter,
                                ema_coef=None if cfg.msk_config.ip_pool_ema == -1 else cfg.msk_config.ip_pool_ema,
                            )
                        ).copy
                    )
                )
                sub_flops: defaultdict[tuple[int, ...], float | None] = field(
                    default_factory=lambda: defaultdict(types.NoneType)
                )
            else:
                prune_metric: defaultdict[str, AverageMeter] = field(default_factory=lambda: defaultdict(AverageMeter))
                sub_flops: float = field(default=1.0)

        if cfg.msk_config.extreme_merge:
            __states__ += ("sub_gps",)
            sub_gps: list["MaskedWidthProxySpaceMixIn.DependencyGroup"] = field(default_factory=list)

        @property
        def device(self) -> torch.device:
            return next(iter(self.layers.values())).weight.device

        def to(self, device: Optional[torch.device] = None):
            if device is None:
                device = self.device

            if self.channel_nums is not None:
                self.channel_nums = self.channel_nums.to(device)
            if self.l1_order is not None:
                self.l1_order = self.l1_order.to(device)
            if self.order_initialized is not None:
                self.order_initialized = self.order_initialized.to(device)
            if self.mask is not None:
                # Avoid directly replacing the mask in case something may lose the reference to it.
                self.mask.update((idx, mask.to(device)) for idx, mask in self.mask.items())
            if self.ch_divisor is not None:
                self.ch_divisor = self.ch_divisor.to(device)

        @classmethod
        def set_optimizer(cls, optimizer: torch.optim.Optimizer):
            cls.__optimizer = optimizer

        @classmethod
        def set_auto_add(cls, auto_add: bool):
            cls.__auto_add = auto_add

        @property
        def repeat_nums(self):
            return torch.diff(self.channel_nums, prepend=torch.zeros_like(self.channel_nums[0:1]))

        def init_mask(self, style: Literal["zeros", "gaussian"] = "gaussian") -> Tensor:
            # TODO: Compatibility with `extreme_merge`.
            if style == "zeros":
                mask = torch.zeros(len(self.channel_nums), dtype=torch.float32)
            elif style == "gaussian":
                mask = torch.empty(len(self.channel_nums), dtype=torch.float32)
                torch.nn.init.normal_(mask, std=cfg.msk_config.init_std)
            else:
                raise ValueError(f"Unknown mask initialization style: {style}.")
            return mask

        def get_mask(self, mask_idx: int, style: Literal["zeros", "gaussian"] = "gaussian"):
            try:
                return self.mask[mask_idx]
            except KeyError:
                if self.__auto_add:
                    self.add_mask(mask_idx, self.init_mask(style), self.__optimizer)
                    return self.mask[mask_idx]
                else:
                    raise

        def add_mask(
            self,
            idx: int,
            mask: Tensor,
            optimizer: torch.optim.Optimizer,
            device: Optional[torch.device] = None,
        ):
            # As the `DependencyGroup` is not under the hood of DDP,
            # we need to broadcast the mask manually when adding it to the optimizer.
            mask = nn.Parameter(
                broadcast_tensor(
                    mask.to(self.device if device is None else device),
                    0,
                    is_ddp=cfg.is_ddp,
                )
            )
            self.mask[idx] = mask
            try:
                next(
                    filter(
                        lambda pg: "idx" in pg and pg["idx"] == idx,
                        optimizer.param_groups,
                    )
                )["params"].append(mask)
            except StopIteration:
                optimizer.add_param_group({"idx": idx, "params": mask})

        def remove_mask(self, idx: int, optimizer: torch.optim.Optimizer):
            del self.mask[idx]
            optimizer.param_groups.remove(next(filter(lambda pg: pg["idx"] == idx, optimizer.param_groups)))

    @dataclass(frozen=True, order=True)
    class ForwardMeta(ForwardMeta):
        soft_in_channels: Optional[Tensor] = field(default=None)
        mask: Optional[Tensor] = field(default=None)
        conv_name: Optional[str] = field(default=None)

    class ReLU(nn.ReLU): ...

    Linear = BasicWidthProxyMixIn.Linear
    SyncBatchNorm = BasicWidthProxyMixIn.SyncBatchNorm

    class Conv2d(BasicWidthProxyMixIn.Conv2d):
        _name: str

        def _get_oc_rep_sigmoid(
            self,
            input: Tensor,
            in_meta: ForwardMeta,
            dep_info: DependencyInfo,
        ) -> tuple[Tensor, Tensor]:
            """
            Return a representative for output channels in the form of `Tensor` and the corresponding output indices.
            """
            ...

    class STECeil(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            i = torch.ceil(args[0])
            ch_divisor = args[1]
            max_ch = args[2]
            try:
                o = torch.zeros_like(i).copy_(min(make_divisible(i.item(), divisor=ch_divisor), max_ch))
            except TypeError:
                # For compatibility with PyTorch 1.x
                o = torch.zeros_like(i)
                tmp = torch.tensor(
                    min(make_divisible(i.item(), divisor=ch_divisor), max_ch),
                    dtype=o.dtype,
                    device=o.device,
                )
                o.copy_(tmp)

            return o

        @staticmethod
        def backward(ctx, *grad_outputs):
            return grad_outputs[0], None, None

    class MaskOptimizer(torch.optim.SGD):
        __flops_loss: float
        __dis_loss: float
        __ctx: "MaskedWidthProxySpaceMixIn"
        state: Dict[str | nn.Parameter, Any]

        if cfg.msk_config.auto_lr:

            @dataclass
            class AutoLRStates:
                auto_lr: float
                cached_arch: Optional[tuple] = None
                interval_counter: int = 0
                pre_remain_counter: int = 0
                variant_counter: int = 0
                remain_counter: int = 0
                explore_end: bool = False
                lr_fixed: bool = False
                hit_fp_target: bool = False
                init_fp_loss: Optional[float] = None
                init_all_grad_norm: Optional[float] = None

        def __init__(
            self,
            params=None,
            lr=cfg.lr,
            momentum=cfg.momentum,
            dampening=0,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        ):
            # As parameters are likely to be appended later, just initialize a dummy one.
            init_params = [torch.nn.Parameter(torch.empty(0))] if params is None else params
            super().__init__(init_params, lr, momentum, dampening, weight_decay, nesterov)
            self.param_groups.pop(0) if params is None else None

            if cfg.msk_config.auto_lr:
                (
                    min_lr,
                    max_lr,
                    iters,
                    threshold,
                    pre_remain_num,
                    variant_num,
                    remain_num,
                ) = cfg.msk_config.auto_lr_config

                self.state["auto_lr_states"] = defaultdict(partial(self.AutoLRStates, min_lr))
                self.auto_lr_scale_factor: float = (max_lr / min_lr) ** (1 / iters)
                self.max_auto_lr: float = max_lr
                self.auto_lr_threshold: float = threshold
                self.pre_max_remain_number = pre_remain_num
                self.min_variant_num: float = variant_num
                self.max_remain_num: float = remain_num
                self.auto_lr_flops_threshold: float = cfg.msk_config.flops_loss_coef * (
                    (cfg.msk_config.flops_relax + cfg.msk_config.auto_lr_flops_relax) ** 2
                )

        @classmethod
        def set_ctx(cls, ctx: "MaskedWidthProxySpaceMixIn"):
            cls.__ctx = ctx

        @staticmethod
        @torch.no_grad()
        def _infer_msk_lr(
            msk: Tensor,
            msk_norm: Tensor,
            max_norm: Tensor,
            delta_msk: Tensor,
            ch_nums: Tensor,
            k: float,
            method: Literal["default"] = "default",
            f_rtol: float = 0.1,
            k_relax: float = 2.0,
            maxiter: int = 20,
        ):
            assert k > 0
            match method:
                case "default":
                    msk_softmax = msk.softmax(0)
                    j_ms = torch.diag(msk_softmax) - msk_softmax.unsqueeze(1) @ msk_softmax.unsqueeze(0)
                    taylor_1_coef = torch.dot(j_ms @ delta_msk, ch_nums)

                    if taylor_1_coef == 0:
                        return ("failed",)

                    elif taylor_1_coef > 0:
                        k = -k

                    if msk_norm > max_norm:
                        normed_msk = msk * max_norm / msk_norm
                        norm_msk_softmax = normed_msk.softmax(0)
                        k = k - torch.dot(norm_msk_softmax - msk_softmax, ch_nums)

                        if k == 0:
                            return "success", normed_msk, 0, 0, 0

                        msk, msk_softmax = normed_msk, norm_msk_softmax
                        j_ms = torch.diag(msk_softmax) - msk_softmax.unsqueeze(1) @ msk_softmax.unsqueeze(0)
                        taylor_1_coef = torch.dot(j_ms @ delta_msk, ch_nums)

                    cur_ch = torch.dot(msk_softmax, ch_nums)
                    # noinspection PyTypeChecker
                    relax_ch = max(1, (k_relax * k).abs())
                    if (k < 0 and cur_ch < (ch_nums[0] + relax_ch)) or (k > 0 and cur_ch > (ch_nums[-1] - relax_ch)):
                        return "success", msk, 0, 0, 0

                    x = k / taylor_1_coef
                    new_msk = msk + x * delta_msk
                    ch_var = torch.dot(new_msk.softmax(0) - msk_softmax, ch_nums)
                    rel_err = torch.abs(torch.sub(ch_var, k) / k)

                    if rel_err > f_rtol:
                        f_tol = (k * f_rtol).abs()
                        try:
                            x = rootfinder(
                                lambda _x: torch.sub(
                                    torch.dot(
                                        (msk + _x * delta_msk).softmax(0) - msk_softmax,
                                        ch_nums,
                                    ),
                                    k,
                                ),
                                x,
                                method="linearmixing",
                                x_tol=float("inf"),
                                f_tol=f_tol,
                                maxiter=maxiter,
                            )
                        except (ConvergenceWarning, ValueError):
                            return ("failed",)

                        new_msk = msk + x * delta_msk
                        ch_var = torch.dot(new_msk.softmax(0) - msk_softmax, ch_nums)

                    return "success", new_msk, x, k, ch_var

                case _:
                    raise NotImplementedError

        @staticmethod
        def _get_rand_bal() -> float:
            init_mean, end_mean, var, trans_epoch = cfg.msk_config.rand_bal_config
            rand_type, trans_type = (
                cfg.msk_config.rand_bal_type,
                cfg.msk_config.rand_bal_trans_type,
            )

            match trans_type:
                case "linear":
                    cur_step = min(max((cfg.rt_epoch - cfg.st_warmup_epochs) / trans_epoch, 0), 1)
                    mean = cur_step * (end_mean - init_mean) + init_mean
                case _:
                    raise NotImplementedError

            match rand_type:
                case "fixed":
                    return mean
                case "gamma":
                    beta = mean / var
                    alpha = mean * beta
                    rand_dist = torch.distributions.Gamma(alpha, beta)
                    return rand_dist.sample().item()
                case "uniform":
                    return (end_mean - init_mean) * torch.rand(1).item() + init_mean
                case _:
                    raise NotImplementedError

        @staticmethod
        def _cal_min_norm(in_tensor: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Returns:
                Tuple[The minimum 2-norm of `in_tensor + const`, and the corresponding `const`]
            """
            a, b, c = len(in_tensor), 2 * sum(in_tensor), sum(in_tensor**2)
            return cast(
                Tuple[Tensor, Tensor],
                (torch.sqrt((4 * a * c - b**2) / (4 * a)), -b / (2 * a)),
            )

        @functools.cache
        def _msk_gt_var(self, length: int, target: float) -> float:
            tmp = torch.zeros(length, device=f"cuda:{cfg.device}", dtype=torch.float32)
            tmp[0] = target
            tmp[1:] = (1 - target) / (length - 1)
            tmp = torch.log(tmp)  # Inverse the softmax
            return torch.var(tmp).item()

        def _get_group(self, mask_idx: int):
            return next(filter(lambda gp: gp["idx"] == mask_idx, self.param_groups))

        def prep_step(self, flops_loss: Tensor | float, dis_loss: Tensor | float):
            """
            Temporarily store the current FLOPs loss and distillation loss.
            """
            self.__flops_loss = flops_loss.item() if isinstance(flops_loss, Tensor) else flops_loss
            self.__dis_loss = dis_loss.item() if isinstance(dis_loss, Tensor) else dis_loss

        def store_grad(self, mask_idx: int, store_name: str):
            def _(_p: nn.Parameter):
                self.state[_p][store_name] = _p.grad.detach().clone() if _p.grad is not None else None
                _p.grad = None

            list(map(_, self._get_group(mask_idx)["params"]))

        def momentum_grad(
            self,
            p: nn.Parameter,
            d_p: Tensor,
            d_p_name: str,
            momentum: float,
            dampening: float,
            nesterov: bool,
        ):
            if momentum != 0:
                buf = self.state[p].get(f"{d_p_name}_buf", None)

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    self.state[p][f"{d_p_name}_buf"] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            return d_p

        @lru_cache(maxsize=1)
        def norm_const(self) -> float:
            max_norm: float = cfg.msk_config.max_norm
            st_warmup_epoch: int = cfg.st_warmup_epochs
            max_norm_epoch: int = min(cfg.msk_config.hard_ft_epoch, cfg.epochs)
            dataset_len = len(cfg.rt_train_loader)
            max_norm_iter = (max_norm_epoch - st_warmup_epoch) * dataset_len
            return max_norm / max_norm_iter

        def dis_buf_update(
            self,
            p: nn.Parameter,
            momentum: float,
            dampening: float,
            nesterov: bool,
        ):
            dis_grad = p.grad
            if dis_grad is not None:
                p.grad = self.momentum_grad(p, dis_grad, "dis", momentum, dampening, nesterov)

        def flops_buf_update(
            self,
            p: nn.Parameter,
            momentum: float,
            dampening: float,
            nesterov: bool,
        ):
            flops_grad = self.state[p]["flops_grad"]
            if flops_grad is not None:
                self.state[p]["flops_grad"] = self.momentum_grad(p, flops_grad, "flops", momentum, dampening, nesterov)

        def sup_soft_buf_update(
            self,
            p: nn.Parameter,
            momentum: float,
            dampening: float,
            nesterov: bool,
        ):
            sup_soft_grad = self.state[p].get("sup_soft_grad", None)
            if sup_soft_grad is not None:
                self.state[p]["sup_soft_grad"] = self.momentum_grad(
                    p, sup_soft_grad, "sup_soft", momentum, dampening, nesterov
                )

        def buf_update(
            self,
            p: nn.Parameter,
            momentum: float,
            dampening: float,
            nesterov: bool,
        ):
            # Manually handle the synchronization of the gradients.
            # TODO: Move the synchronization to a more appropriate place.
            dis_grad = p.grad
            if dis_grad is not None:
                p.grad = all_reduce_avg(dis_grad, cfg.is_ddp)

            if cfg.msk_config.decoupled_kl:
                sup_soft_grad = self.state[p].get("sup_soft_grad", None)
                if sup_soft_grad is not None:
                    self.state[p]["sup_soft_grad"] = all_reduce_avg(sup_soft_grad, cfg.is_ddp)

            # TODO: Better handling the buffer update when using `decoupled_kl`.
            match cfg.msk_config.buffer_type:
                case "none" | "grad":
                    pass
                case "dis":
                    self.dis_buf_update(p, momentum, dampening, nesterov)
                    if cfg.msk_config.decoupled_kl:
                        self.sup_soft_buf_update(p, momentum, dampening, nesterov)
                case "flops":
                    self.flops_buf_update(p, momentum, dampening, nesterov)
                case "dis_flops":
                    self.dis_buf_update(p, momentum, dampening, nesterov)
                    if cfg.msk_config.decoupled_kl:
                        self.sup_soft_buf_update(p, momentum, dampening, nesterov)
                    self.flops_buf_update(p, momentum, dampening, nesterov)
                case _:
                    raise NotImplementedError

            if cfg.msk_config.decoupled_kl:
                dis_grad = p.grad
                sup_soft_grad = self.state[p].get("sup_soft_grad", None)
                match dis_grad, sup_soft_grad:
                    case _, None:
                        pass
                    case None, _:
                        p.grad = sup_soft_grad
                    case _, _:
                        dis_norm = torch.linalg.norm(dis_grad.flatten(), ord=2)
                        sup_soft_norm = torch.linalg.norm(sup_soft_grad.flatten(), ord=2)
                        if cfg.msk_config.rand_bal and cfg.msk_config.dkl_sync_rand_bal:
                            trans_epoch = cfg.msk_config.rand_bal_config[-1]
                            cur_step = min(
                                max(
                                    (cfg.rt_epoch - cfg.st_warmup_epochs) / trans_epoch,
                                    0,
                                ),
                                1,
                            )
                            sup_coef, soft_coef = 1 - cur_step, cur_step
                            p.grad = sup_soft_grad * (sup_coef / sup_soft_norm) + dis_grad * (soft_coef / dis_norm)
                        elif (sf := cfg.msk_config.sup_soft_dis_scale) == float("inf"):
                            p.grad = sup_soft_grad
                        elif sf == 0:
                            p.grad = dis_grad
                        else:
                            sup_soft_grad = sup_soft_grad * (dis_norm / sup_soft_norm * sf)
                            p.grad = dis_grad + sup_soft_grad

        @torch.no_grad()
        def cus_step(
            self,
            mask_idx: int,
            mode: Literal["gen_pool", "sub_optim"],
            pgs: float | None = None,
            cus_update: bool | None = None,
        ):
            """
            Replacing the original `step` with the customized interface.
            """

            def _(i, p: nn.Parameter):
                nonlocal lr

                # TODO: Is it necessary to synchronize the FLOPs gradients?
                flops_grad = self.state[p]["flops_grad"]
                # TODO: Better handling the `None` case of all data-independent losses,
                #  currently, the FLOPs is dominant, if FLOPs gradient is None,
                #  we do not care about other data-independent losses.
                if flops_grad is not None:
                    has_norm = (norm_grad := self.state[p].get("norm_grad", None)) is not None
                    has_entropy = (entropy_grad := self.state[p].get("entropy_grad", None)) is not None

                    flops_norm = None
                    if has_norm or has_entropy:
                        flops_norm = torch.linalg.norm(flops_grad.flatten(), ord=2)

                    if has_norm:
                        # Balancing the gradients of the FLOPs loss and the norm loss
                        norm_norm = torch.linalg.norm(norm_grad.flatten(), ord=2)
                        if norm_norm > 0:
                            scale_factor = cfg.msk_config.flops_norm_scale
                            norm_grad = norm_grad * (flops_norm / norm_norm / scale_factor)
                            flops_grad = flops_grad + norm_grad

                    if has_entropy:
                        # Balancing the gradients of the FLOPs loss and the entropy loss
                        entropy_norm = torch.linalg.norm(entropy_grad.flatten(), ord=2)
                        if entropy_norm > 0:
                            scale_factor = cfg.msk_config.flops_entropy_scale
                            entropy_grad = entropy_grad * (flops_norm / entropy_norm / scale_factor)
                            flops_grad = flops_grad + entropy_grad

                # TODO: Is it necessary to synchronize the parameters?
                # p.copy_(broadcast_tensor(p, 0, cfg.is_ddp))
                dis_grad = p.grad

                # Retrieve the balance factor
                if cfg.msk_config.rand_bal:
                    nonlocal dis_flops_scale
                else:
                    dis_flops_scale = cfg.msk_config.dis_flops_scale

                grad = None
                match dis_grad, flops_grad:
                    case None, None:
                        warnings.warn(
                            "No gradient is available for the parameter. The optimization is just skipped.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return
                    case None, _:  # Where there is no distillation loss
                        grad = flops_grad
                    case _, None:  # Where there is no FLOPs loss
                        grad = dis_grad
                    case _, _ if cfg.msk_config.orthogonal:
                        grad = (
                            dis_grad - torch.dot(dis_grad, flops_grad) / torch.dot(flops_grad, flops_grad) * flops_grad
                        )
                    case _, _ if cfg.msk_config.equal_msk_update:
                        # Normalize the gradient of the FLOPs loss to the one of the distillation loss
                        dis_norm = torch.linalg.norm(dis_grad.flatten(), ord=2)
                        flops_norm = torch.linalg.norm(flops_grad.flatten(), ord=2)

                        if flops_norm > 0:
                            flops_grad = flops_grad * (dis_norm / flops_norm / dis_flops_scale)

                        grad = dis_grad if flops_loss < flops_threshold else flops_grad
                        # grad = dis_grad + flops_grad
                    case _, _:
                        # Normalize the gradient of the FLOPs loss to the one of the distillation loss
                        dis_norm = torch.linalg.norm(dis_grad.flatten(), ord=2)
                        flops_norm = torch.linalg.norm(flops_grad.flatten(), ord=2)

                        # For case where dis_grad is large,
                        # while flops_grad is relatively small and flops loss is high,
                        # FLOPs takes priority, we always want to ensure the FLOPs target.
                        dy_scale = min(flops_loss / flops_threshold, 1.0)
                        if dy_scale != 0:
                            scale_factor = dis_flops_scale / dy_scale

                            match cfg.msk_config.grad_fuse:
                                case "default":
                                    if (dis_norm > flops_norm * scale_factor) and flops_norm > 0:
                                        flops_grad = flops_grad * (dis_norm / flops_norm / scale_factor)
                                    elif (dis_norm < flops_norm * scale_factor) and dis_norm > 0:
                                        dis_grad = dis_grad * (flops_norm / dis_norm * scale_factor)

                                case "align_dis":
                                    cur_gp_len = len(self.__ctx._dep_groups[i])
                                    dis_grad = dis_grad / cur_gp_len
                                    flops_grad = flops_grad * (dis_norm / flops_norm / dis_flops_scale / cur_gp_len)

                                case "align_dis_no_norm":
                                    flops_grad = flops_grad * (dis_norm / flops_norm / dis_flops_scale)

                                case "global_align_dis":
                                    cur_gp_len = len(self.__ctx._dep_groups[i])
                                    dis_grad = dis_grad * (1.0 / cur_gp_len / all_dis_grad_norm)
                                    flops_grad = flops_grad * (
                                        dis_norm / flops_norm / dis_flops_scale / cur_gp_len / all_dis_grad_norm
                                    )

                                case "global_align_dis_v2":
                                    cur_gp_len = len(self.__ctx._dep_groups[i])
                                    additional_norm_term = cur_gp_len * all_dis_grad_norm / math.sqrt(all_gp_len)
                                    dis_grad = dis_grad / additional_norm_term
                                    flops_grad = flops_grad * (
                                        dis_norm / flops_norm / dis_flops_scale / additional_norm_term
                                    )

                                case "global_align_dis_no_norm":
                                    dis_grad = dis_grad / all_dis_grad_norm
                                    flops_grad = flops_grad * (
                                        dis_norm / flops_norm / dis_flops_scale / all_dis_grad_norm
                                    )

                                case "slow_norm":
                                    flops_grad = flops_grad * (dis_norm / flops_norm / scale_factor)

                                case "fp_first":
                                    if dy_scale >= 1:
                                        dis_grad = dis_grad * (flops_norm / dis_norm * scale_factor)
                                    else:
                                        flops_grad = flops_grad * (dis_norm / flops_norm / scale_factor)

                                case "fp_only" | "post_fp_only":
                                    dis_grad = dis_grad * (flops_norm / dis_norm * scale_factor)

                                case "bal_only" | "post_bal_only" | "post_bal_only_v2":
                                    dis_grad = dis_grad * (flops_norm / dis_norm * dis_flops_scale)

                                case "bal_only_v2":
                                    if dis_norm != 0:
                                        dis_grad = dis_grad * (flops_norm / dis_norm * dis_flops_scale)

                                    fp_norm_2 = flops_norm**2
                                    if fp_norm_2 != 0:
                                        lr = (
                                            2
                                            * torch.sqrt(
                                                torch.mul(
                                                    flops_loss,
                                                    cfg.msk_config.flops_loss_coef,
                                                )
                                            )
                                            * cfg.msk_config.emu_coef
                                            * min(min_fp_grad_norm, flops_norm)
                                            / len(self.__ctx._dep_groups)
                                            / fp_norm_2
                                        )
                                    else:
                                        lr = 0

                                    if cfg.msk_config.auto_lr:
                                        assert cfg.msk_config.auto_lr_schedule_only
                                        lr *= auto_lr_decay

                                case "global_bal_only":
                                    dis_grad = dis_grad * (all_fp_grad_norm / all_dis_grad_norm * dis_flops_scale)

                                case "global_bal_only_v2":
                                    if all_dis_grad_norm != 0:
                                        dis_grad = dis_grad * (all_fp_grad_norm / all_dis_grad_norm * dis_flops_scale)

                                    fp_norm_2 = flops_norm**2
                                    if fp_norm_2 != 0:
                                        lr = (
                                            2
                                            * torch.sqrt(
                                                torch.mul(
                                                    flops_loss,
                                                    cfg.msk_config.flops_loss_coef,
                                                )
                                            )
                                            * cfg.msk_config.emu_coef
                                            * min(min_fp_grad_norm, flops_norm)
                                            / len(self.__ctx._dep_groups)
                                            / fp_norm_2
                                        )
                                    else:
                                        lr = 0

                                    if cfg.msk_config.auto_lr:
                                        assert cfg.msk_config.auto_lr_schedule_only
                                        lr *= auto_lr_decay

                                case "global_bal_only_v3":
                                    # if all_dis_grad_norm != 0:
                                    # dis_grad = dis_grad * (all_fp_grad_norm / all_dis_grad_norm * dis_flops_scale)
                                    dis_grad = 0

                                    # noinspection PyUnresolvedReferences
                                    gp_flops = sum(
                                        map(
                                            operator.itemgetter(1),
                                            self.__ctx._dep_groups[i].gp_flops.values(),
                                        )
                                    )
                                    gp_ratio = gp_flops / all_gp_flops
                                    fp_norm_2 = flops_norm**2
                                    if fp_norm_2 != 0:
                                        lr = (
                                            2
                                            * torch.sqrt(
                                                torch.mul(
                                                    flops_loss,
                                                    cfg.msk_config.flops_loss_coef,
                                                )
                                            )
                                            * cfg.msk_config.emu_coef
                                            # * min(min_fp_grad_norm, flops_norm)
                                            * gp_ratio
                                            / fp_norm_2
                                        )
                                    else:
                                        lr = 0

                                    if cfg.msk_config.auto_lr:
                                        assert cfg.msk_config.auto_lr_schedule_only
                                        lr *= auto_lr_decay

                                case "fp_same_norm_bal_only":
                                    # TODO: What if the order of dependency groups
                                    #  mismatches the order of the optimizer param_groups?
                                    # noinspection PyUnresolvedReferences
                                    cur_gp_len = len(self.__ctx._dep_groups[i].channel_nums)
                                    unified_norm = all_fp_grad_norm * math.sqrt(cur_gp_len / all_gp_len)
                                    flops_grad = flops_grad * unified_norm / flops_norm
                                    dis_grad = dis_grad * (unified_norm / dis_norm * dis_flops_scale)

                                case "global_fp_same_norm_bal_only":
                                    # TODO: What if the order of dependency groups
                                    #  mismatches the order of the optimizer param_groups?
                                    # noinspection PyUnresolvedReferences
                                    cur_gp_len = len(self.__ctx._dep_groups[i].channel_nums)
                                    unified_norm = all_fp_grad_norm * math.sqrt(cur_gp_len / all_gp_len)
                                    flops_grad = flops_grad * unified_norm / flops_norm
                                    dis_grad = dis_grad * (all_fp_grad_norm / all_dis_grad_norm * dis_flops_scale)

                                case "no_dis":
                                    dis_grad = 0

                                case "no_flops":
                                    flops_grad = 0

                                case _:
                                    raise NotImplementedError(
                                        f"Unsupported gradient fuse mode: {cfg.msk_config.grad_fuse}!"
                                    )

                        grad = dis_grad + flops_grad

                        match cfg.msk_config.grad_rescale:
                            case "none":
                                pass
                            case "dis_norm":
                                grad_norm = torch.linalg.norm(grad.flatten(), ord=2)
                                if grad_norm != 0:
                                    grad = grad * (dis_norm / grad_norm)
                            case "flops_norm":
                                grad_norm = torch.linalg.norm(grad.flatten(), ord=2)
                                if grad_norm != 0:
                                    grad = grad * (flops_norm / grad_norm)
                            case _:
                                raise NotImplementedError

                if cfg.msk_config.grad_fuse in (
                    "post_bal_only",
                    "post_bal_only_v2",
                    "post_fp_only",
                ):
                    assert not cfg.msk_config.equal_msk_update
                    self.state[p]["tmp_grad"] = grad
                    return

                if cfg.msk_config.equal_msk_update:
                    nonlocal update_enabled

                    if self.state[p].get("update_buffer", []):
                        return

                    p_norm = torch.linalg.norm(p, ord=2)
                    grad_norm = torch.linalg.norm(grad, ord=2)
                    if grad_norm == 0:
                        return

                    fixed_dy_lr = self.state[p].get("fixed_dy_lr", None)
                    if flops_loss < flops_threshold and fixed_dy_lr is None:
                        fixed_dy_lr = 0.01 * self.state[p].get("last_dy_lr", 1)
                        self.state[p]["fixed_dy_lr"] = fixed_dy_lr
                        del self.state[p]["last_dy_lr"]

                    p_norm_grad = grad * (p_norm / grad_norm)
                    if fixed_dy_lr is not None:
                        p.add_(p_norm_grad, alpha=fixed_dy_lr)
                        update_enabled = False
                        return

                    k = cfg.msk_config.emu_coef

                    # noinspection PyUnresolvedReferences
                    ch_nums = self.__ctx._dep_groups[i].channel_nums.float()
                    max_ch = ch_nums[-1]
                    k_ori = k * max_ch
                    k = self.state[p].get("k_scale", 1) * k_ori

                    # Norm constraint
                    assert cfg.msk_config.max_init
                    max_norm = self.state[p].get("max_norm", None)
                    if max_norm is None:
                        max_norm = 2 * p_norm
                        self.state[p]["max_norm"] = max_norm

                    failed_count = self.state[p].get("failed_count", 0)
                    # noinspection PyTypeChecker
                    status, *result = self._infer_msk_lr(
                        p,
                        p_norm,
                        max_norm,
                        p_norm_grad,
                        ch_nums,
                        k,
                        cfg.msk_config.emu_solver,
                        maxiter=10 * (failed_count + 1),
                    )

                    if status == "success":
                        new_p, dy_lr, k, ch_var = result
                        if torch.any(torch.isnan(new_p)):
                            print("OK")
                        sorted_p = new_p.sort()[0]
                        bias = 0.5 * (sorted_p[-1] + sorted_p[0])
                        new_p.add_(-bias)
                        p_norm_after = torch.linalg.norm(new_p, ord=2)
                        p_norm_delta = p_norm_after - p_norm
                        imp_score = torch.dot(
                            dis_grad,
                            p_norm_grad * ((dy_lr * k_ori / k) if k != 0 else 0),
                        )
                        self.state[p]["update_buffer"] = [
                            new_p,
                            k,
                            dy_lr,
                            ch_var,
                            p_norm_delta,
                            imp_score,
                        ]
                        self.state[p]["failed_count"] = 0
                        if dy_lr != 0:
                            assert k != 0
                            self.state[p]["last_dy_lr"] = -((dy_lr * k_ori / k).abs())

                    else:
                        assert status == "failed"
                        self.state[p]["failed_count"] = failed_count + 1

                    if self.state[p].get("failed_count", 0) > 0:
                        update_enabled = False

                elif cus_update == "var":
                    assert not cfg.msk_config.auto_lr
                    p_norm = torch.linalg.norm(p.flatten(), ord=2)
                    grad_norm = torch.linalg.norm(grad.flatten(), ord=2)
                    scale_factor = cfg.msk_config.param_grad_scale if pgs is None else pgs
                    lr_decay = lr / init_lr if cfg.msk_config.lr_decay else 1.0

                    if p_norm > grad_norm * scale_factor:
                        grad = grad * (p_norm / grad_norm / scale_factor)

                    new_p = p - grad * lr_decay

                    oht = cfg.msk_config.one_hot_target if mode == "sub_optim" else (1 - cfg.msk_config.one_hot_supp)
                    gt_var = self._msk_gt_var(len(new_p), oht)
                    new_p_var = torch.var(new_p)
                    new_p = new_p * torch.sqrt(gt_var / new_p_var)
                    new_p = new_p + self._cal_min_norm(new_p)[1]

                    p.copy_(new_p)

                elif cus_update == "norm":
                    assert not cfg.msk_config.auto_lr
                    grad_f = grad.flatten()
                    grad_norm = torch.linalg.norm(grad_f, ord=2)

                    if grad_norm == 0:
                        warnings.warn(
                            "The gradient is zero, the optimization is just skipped. "
                            "It is not expected to happen. Please debug into this issue.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return

                    p_f = p.flatten()
                    p_norm = torch.linalg.norm(p_f, ord=2)
                    p_dot_grad = torch.dot(p_f, grad_f)
                    # TODO: How to determine the `norm_const` for each parameter,
                    #  and how to update it so that the increased norm can be smaller when reaching some norm threshold.
                    norm_const = self.norm_const()
                    # For each update of `p`, we hope the increase of the norm of `p` is just `norm_const`.
                    dy_lr = (
                        p_dot_grad
                        + torch.sqrt(p_dot_grad**2 + grad_norm**2 * (norm_const**2 + 2 * norm_const * p_norm))
                    ) / (grad_norm**2)
                    p.add_(grad, alpha=-dy_lr)

                else:
                    assert cus_update == "default"

                    if torch.isnan(grad).any():
                        warnings.warn(
                            f"NaN detected while optimizing the mask, although recognized as zero gradient, "
                            f"it is recommended to debug into this issue at iteration {cfg.rt_iter}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        grad.zero_()

                    if cfg.msk_config.buffer_type == "grad":
                        grad = self.momentum_grad(p, grad, "grad", momentum, dampening, nesterov)

                    if weight_decay != 0:
                        grad = grad.add(p, alpha=weight_decay)

                    p.add_(grad, alpha=-lr)

                if cfg.msk_config.keep_min_norm:
                    match cfg.msk_config.min_norm_type:
                        case 2:
                            p.add_(self._cal_min_norm(p)[1])
                        case "inf":
                            sorted_mask = p.sort()[0]
                            bias = 0.5 * (sorted_mask[-1] + sorted_mask[0])
                            p.add_(-bias)
                        case _:
                            raise NotImplementedError

                p.grad = None
                del self.state[p]["flops_grad"]
                if "sup_soft_grad" in self.state[p]:
                    del self.state[p]["sup_soft_grad"]

            if cus_update is None:
                cus_update = cfg.msk_config.cus_update

            if cfg.msk_config.equal_msk_update:
                update_enabled = True

            flops_threshold = self.__ctx._flops_threshold
            flops_loss = self.__flops_loss

            cur_group = self._get_group(mask_idx)
            ref_group = cfg.rt_optimizer.param_groups[0]

            momentum = ref_group.get("momentum", 0.9)
            dampening = ref_group.get("dampening", 0)
            nesterov = ref_group.get("nesterov", True)
            weight_decay = (
                ref_group["weight_decay"]
                if cfg.msk_config.msk_weight_decay == "same"
                else cfg.msk_config.msk_weight_decay
            )
            lr = ref_group["lr"]
            init_lr = ref_group["initial_lr"]

            # Equal-scale reduction for the given mask learning rate.
            msk_lr = cfg.msk_config.msk_lr

            # TODO: `t4` norm based on total parameters of layers in a dependency group.
            if cfg.msk_config.gp_grad_norm == "t3":
                # FLOPs/Distillation gradient normalization.
                # TODO: What if the order of dependency groups mismatches the order of the optimizer param_groups?
                for gp, param in zip(self.__ctx._dep_groups, cur_group["params"]):
                    self.state[param]["flops_grad"].div_(len(gp)) if self.state[param][
                        "flops_grad"
                    ] is not None else None
                    if cfg.msk_config.grad_fuse in (
                        "global_bal_only",
                        "global_fp_same_norm_bal_only",
                    ):
                        param.grad.div_(len(gp)) if param.grad is not None else None

            for param in cur_group["params"]:
                self.buf_update(param, momentum, dampening, nesterov)

            if cfg.msk_config.grad_fuse == "global_bal_only_v3":
                # noinspection PyUnresolvedReferences
                all_gp_flops = sum(
                    sum(map(operator.itemgetter(1), gp.gp_flops.values())) for gp in self.__ctx._dep_groups
                )

            if cfg.msk_config.grad_fuse in (
                "global_bal_only",
                "global_bal_only_v2",
                "global_bal_only_v3",
                "bal_only_v2",
                "fp_same_norm_bal_only",
                "global_fp_same_norm_bal_only",
            ):
                all_fp_grad = list(
                    map(
                        torch.flatten,
                        filter(
                            lambda pg: pg is not None,
                            map(
                                lambda p: self.state[p]["flops_grad"],
                                cur_group["params"],
                            ),
                        ),
                    )
                )
                if not all_fp_grad:
                    raise RuntimeError("No FLOPs gradient is available for balancing the global gradient.")
                all_fp_grad_norm = torch.linalg.norm(torch.cat(all_fp_grad), ord=2)

                if cfg.msk_config.grad_fuse in (
                    "global_bal_only_v2",
                    "global_bal_only_v3",
                    "bal_only_v2",
                ):
                    fp_grad_norm_list = torch.sort(torch.stack([torch.linalg.norm(_fg, ord=2) for _fg in all_fp_grad]))
                    fp_grad_norm_list = fp_grad_norm_list[0]
                    converge_threshold = (
                        cfg.msk_config.emu_speed
                        * max(
                            math.sqrt(flops_loss / cfg.msk_config.flops_loss_coef) - cfg.msk_config.flops_relax,
                            0,
                        )
                        / (len(cfg.rt_train_loader) * (cfg.epochs - cfg.rt_epoch))
                    )
                    c_idx = torch.searchsorted(fp_grad_norm_list * cfg.msk_config.emu_coef, converge_threshold)

                    if c_idx == len(fp_grad_norm_list):
                        warnings.warn(
                            "The minimum FLOPs gradient norm is too small, the convergence may be affected. "
                            "It is recommended to increase the `emu_coef`.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        c_idx -= 1

                    min_fp_grad_norm = fp_grad_norm_list[c_idx]

            if cfg.msk_config.grad_fuse in (
                "global_bal_only",
                "global_bal_only_v2",
                "global_bal_only_v3",
                "global_fp_same_norm_bal_only",
                "global_align_dis",
                "global_align_dis_v2",
                "global_align_dis_no_norm",
            ):
                all_dis_grad = list(
                    map(
                        torch.flatten,
                        filter(
                            lambda pg: pg is not None,
                            map(
                                lambda p: p.grad,
                                cur_group["params"],
                            ),
                        ),
                    )
                )
                if not all_dis_grad:
                    raise RuntimeError("No distillation gradient is available for balancing the global gradient.")
                all_dis_grad_norm = torch.linalg.norm(torch.cat(all_dis_grad), ord=2)

            if cfg.msk_config.grad_fuse in (
                "fp_same_norm_bal_only",
                "global_fp_same_norm_bal_only",
                "post_bal_only",
                "post_bal_only_v2",
                "post_fp_only",
                "global_align_dis_v2",
            ):
                all_gp_len = sum(map(lambda _gp: len(_gp.channel_nums), self.__ctx._dep_groups))

            if cfg.msk_config.auto_lr:
                state_cls = self.AutoLRStates
                auto_lr_states: defaultdict[int, state_cls] = self.state["auto_lr_states"]
                cur_states = auto_lr_states[mask_idx]
                if cur_states.init_fp_loss is None:
                    cur_states.init_fp_loss = flops_loss
                    if cfg.msk_config.grad_fuse in (
                        "slow_norm",
                        "align_dis",
                        "align_dis_no_norm",
                        "global_align_dis",
                        "global_align_dis_v2",
                        "global_align_dis_no_norm",
                    ):
                        all_dis_grad = list(
                            map(
                                torch.flatten,
                                filter(
                                    lambda pg: pg is not None,
                                    map(
                                        lambda p: p.grad,
                                        cur_group["params"],
                                    ),
                                ),
                            )
                        )
                        if not all_dis_grad:
                            raise RuntimeError(
                                f"[{mask_idx}] No distillation gradient is available "
                                f"for the mask learning rate exploration."
                            )
                        all_grad_norm = torch.linalg.norm(torch.cat(all_dis_grad), ord=2)
                    else:
                        all_fp_grad = list(
                            map(
                                torch.flatten,
                                filter(
                                    lambda pg: pg is not None,
                                    map(
                                        lambda p: self.state[p]["flops_grad"],
                                        cur_group["params"],
                                    ),
                                ),
                            )
                        )
                        if not all_fp_grad:
                            raise RuntimeError(
                                f"[{mask_idx}] No FLOPs gradient is available for the mask learning rate exploration."
                            )
                        all_grad_norm = torch.linalg.norm(torch.cat(all_fp_grad), ord=2)
                    cur_states.init_all_grad_norm = all_grad_norm.item()

                if not cur_states.hit_fp_target and flops_loss <= self.auto_lr_flops_threshold:
                    cur_states.hit_fp_target = True

                if not cur_states.lr_fixed and not cfg.msk_config.auto_lr_schedule_only:
                    cur_states.interval_counter += 1
                    if cur_states.interval_counter > self.auto_lr_threshold:
                        cur_states.interval_counter = 0
                        cur_arch = self.__ctx._cur_subnet.proxy.extract()
                        prev_arch, cur_states.cached_arch = (
                            cur_states.cached_arch,
                            cur_arch,
                        )
                        if not cur_states.explore_end:
                            if (
                                cur_arch == prev_arch
                                and not cur_states.hit_fp_target
                                and cur_states.auto_lr < self.max_auto_lr
                            ):
                                cur_states.pre_remain_counter += 1
                                if cur_states.pre_remain_counter >= self.pre_max_remain_number:
                                    cur_states.pre_remain_counter = 0
                                    cur_states.variant_counter = 0
                                    cur_states.auto_lr *= self.auto_lr_scale_factor
                            elif prev_arch is not None:
                                if cur_arch != prev_arch:
                                    cur_states.pre_remain_counter = 0
                                    cur_states.variant_counter += 1
                                if (
                                    cur_states.variant_counter >= self.min_variant_num
                                    or cur_states.hit_fp_target
                                    or cur_states.auto_lr >= self.max_auto_lr
                                ):
                                    if cur_states.auto_lr <= cfg.msk_config.auto_lr_config[0]:
                                        warnings.warn(
                                            f"[{mask_idx}] Automatic mask learning rate exploration ended with the "
                                            f"minimum value, please consider to adapt the configuration.",
                                            RuntimeWarning,
                                            stacklevel=2,
                                        )
                                    if cur_states.auto_lr >= self.max_auto_lr:
                                        warnings.warn(
                                            f"[{mask_idx}] Automatic mask learning rate exploration ended with the "
                                            f"maximum value, please consider to adapt the configuration.",
                                            RuntimeWarning,
                                            stacklevel=2,
                                        )
                                    cfg.rt_logger.info(
                                        f"[{mask_idx}] Automatic mask learning exploration ended with the "
                                        f"learning rate: {cur_states.auto_lr:.6f}."
                                    )
                                    cur_states.explore_end = True
                        else:
                            if cur_states.hit_fp_target:
                                cfg.rt_logger.info(
                                    f"[{mask_idx}] Architecture reaches the FLOPs target, "
                                    f"fixing the mask learning rate to {cur_states.auto_lr:.6f}."
                                )
                                cur_states.lr_fixed = True
                            elif cur_states.auto_lr >= self.max_auto_lr:
                                cfg.rt_logger.info(
                                    f"[{mask_idx}] Architecture reaches the maximum learning rate, "
                                    f"fixing the mask learning rate to {cur_states.auto_lr:.6f}."
                                )
                                cur_states.lr_fixed = True
                            elif cur_arch == prev_arch:
                                cur_states.remain_counter += 1
                                if cur_states.remain_counter >= self.max_remain_num:
                                    cur_states.remain_counter = 0
                                    cur_states.auto_lr *= self.auto_lr_scale_factor
                                    cfg.rt_logger.info(
                                        f"[{mask_idx}] Architecture keeps unchanged, "
                                        f"increasing the mask learning rate to {cur_states.auto_lr:.6f}."
                                    )
                            else:
                                cur_states.remain_counter = 0

                if not cfg.msk_config.auto_lr_schedule_only:
                    msk_lr = cur_states.auto_lr

                    if cur_states.init_all_grad_norm and cfg.msk_config.grad_fuse not in ("global_align_dis_v2",):
                        msk_lr /= cur_states.init_all_grad_norm
                else:
                    msk_lr = init_lr if msk_lr == "same" else msk_lr

                if cur_states.explore_end and not cfg.msk_config.auto_lr_schedule_only:
                    msk_lr = msk_lr * cfg.msk_config.auto_lr_shift

                # Learning rate reduction based on FLOPs loss.
                # How the x-coordinate is distributed:
                #     `log`: Sparse -> Dense
                #     `linear`: Uniform
                # How the `auto_lr_decay_func` decays: exp > none > sin > log
                min_lr_decay = cfg.msk_config.auto_lr_min_decay

                match cfg.msk_config.auto_lr_decay_map:
                    case "log":
                        x = (
                            (
                                (math.log(flops_loss) - math.log(flops_threshold))
                                / (math.log(cur_states.init_fp_loss) - math.log(flops_threshold))
                            )
                            if flops_loss > 0
                            else 0
                        )
                    case "linear":
                        x = (flops_loss - flops_threshold) / (cur_states.init_fp_loss - flops_threshold)
                    case _:
                        raise NotImplementedError

                match cfg.msk_config.auto_lr_decay_func:
                    case "step":
                        auto_lr_decay = 1 if flops_loss > flops_threshold else min_lr_decay
                    case "exp":
                        # noinspection PyTypeChecker
                        auto_lr_decay = min_lr_decay + (1 - min_lr_decay) * (
                            (1 / (math.e - 1)) * math.exp(min(1, max(0, x))) - (1 / (math.e - 1))
                        )
                    case "none":
                        # noinspection PyTypeChecker
                        auto_lr_decay = min_lr_decay + (1 - min_lr_decay) * min(1, max(0, x))
                    case "sin":
                        # noinspection PyTypeChecker
                        auto_lr_decay = min_lr_decay + (1 - min_lr_decay) * math.sin(math.pi / 2 * min(1, max(0, x)))
                    case "log":
                        # noinspection PyTypeChecker
                        auto_lr_decay = min_lr_decay + (1 - min_lr_decay) * math.log(
                            min(1, max(0, x)) * (math.e - 1) + 1
                        )
                    case _:
                        raise NotImplementedError

                msk_lr *= auto_lr_decay

            if isinstance(msk_lr, float):
                lr = (lr / init_lr * msk_lr) if cfg.msk_config.sync_schedule else msk_lr
                init_lr = msk_lr
            else:
                assert msk_lr == "same"

            if cfg.msk_config.rand_bal:
                dis_flops_scale = self._get_rand_bal()

            list(itertools.starmap(_, enumerate(cur_group["params"])))

            if cfg.msk_config.equal_msk_update:
                # noinspection PyUnboundLocalVariable
                if not update_enabled:
                    return

                if cfg.msk_config.emu_polar == "pos":
                    imp_scores = list(
                        map(
                            lambda _p: self.state[_p].get("update_buffer", [])[-1],
                            cur_group["params"],
                        )
                    )
                else:
                    imp_scores = list(
                        map(
                            lambda _p: -self.state[_p].get("update_buffer", [])[-1],
                            cur_group["params"],
                        )
                    )

                # Linearly map `imp_scores` to [0.5, 2].
                imp_scores = torch.stack(imp_scores)
                imp_min, imp_max = imp_scores.min(), imp_scores.max()
                if imp_min != imp_max:
                    imp_scores = (imp_scores - imp_min) / (imp_max - imp_min) * 1.5 + 0.5
                else:
                    imp_scores = torch.ones_like(imp_scores)

                for i_, p_ in enumerate(cur_group["params"]):
                    update_buffer: list = self.state[p_].get("update_buffer", [])
                    assert update_buffer
                    new_p_, k_, dy_lr_, ch_var_, p_norm_delta_, imp_score_ = update_buffer
                    p_.copy_(new_p_)
                    self.state[p_]["k_scale"] = imp_scores[i_]

                    if cfg.msk_config.emu_tb:
                        cfg.rt_tb_logger.add_scalar(f"imp_score/{i_}", imp_score_, cfg.rt_iter)
                        cfg.rt_tb_logger.add_scalar(f"k_scale/{i_}", imp_scores[i_], cfg.rt_iter)
                        cfg.rt_tb_logger.add_scalar(f"delta_mask_norm/{i_}", p_norm_delta_, cfg.rt_iter)
                        cfg.rt_tb_logger.add_scalar(f"dynamic_lr/{i_}", dy_lr_, cfg.rt_iter)
                        # noinspection PyUnboundLocalVariable
                        cfg.rt_tb_logger.add_scalars(
                            f"delta_hard_arch/{i_}",
                            {"expected": k_, "practical": ch_var_},
                            cfg.rt_iter,
                        )

                    self.state[p_]["update_buffer"] = []

            if cfg.msk_config.grad_fuse in (
                "post_bal_only",
                "post_bal_only_v2",
                "post_fp_only",
            ):
                all_grad = list(
                    map(
                        torch.flatten,
                        filter(
                            lambda pg: pg is not None,
                            map(
                                lambda p: self.state[p]["tmp_grad"],
                                cur_group["params"],
                            ),
                        ),
                    )
                )
                if not all_grad:
                    raise RuntimeError("No gradient is available for post normalization.")
                all_grad_norm = torch.linalg.norm(torch.cat(all_grad), ord=2)

                all_fp_grad = list(
                    map(
                        torch.flatten,
                        filter(
                            lambda pg: pg is not None,
                            map(
                                lambda p: self.state[p]["flops_grad"],
                                cur_group["params"],
                            ),
                        ),
                    )
                )
                if not all_fp_grad:
                    raise RuntimeError("No FLOPs gradient is available for balancing the global gradient.")
                all_fp_grad_norm = torch.linalg.norm(torch.cat(all_fp_grad), ord=2)

                for i_, p_ in enumerate(cur_group["params"]):
                    grad_ = self.state[p_]["tmp_grad"]
                    grad_norm_ = torch.linalg.norm(grad_.flatten(), ord=2)
                    flops_grad_ = self.state[p_]["flops_grad"]
                    flops_norm_ = torch.linalg.norm(flops_grad_.flatten(), ord=2)
                    # TODO: What if the order of dependency groups
                    #  mismatches the order of the optimizer param_groups?
                    if cfg.msk_config.grad_fuse == "post_bal_only":
                        # noinspection PyUnresolvedReferences
                        cur_gp_len_ = len(self.__ctx._dep_groups[i_].channel_nums)
                        # noinspection PyUnboundLocalVariable
                        scale_factor_ = math.sqrt(cur_gp_len_ / all_gp_len)
                        # noinspection PyUnboundLocalVariable
                        grad_ = (
                            grad_
                            * (all_grad_norm * scale_factor_ / grad_norm_)
                            # TODO: Add a configuration for this term.
                            * min((all_fp_grad_norm * scale_factor_ / flops_norm_), 1)
                        )
                    elif cfg.msk_config.grad_fuse == "post_bal_only_v2":
                        scale_factor_ = math.sqrt(1 / len(self.__ctx._dep_groups))
                        grad_ = grad_ * (all_grad_norm * scale_factor_ / grad_norm_)

                    if torch.isnan(grad_).any():
                        warnings.warn(
                            f"NaN detected while optimizing the mask, although recognized as zero gradient, "
                            f"it is recommended to debug into this issue at iteration {cfg.rt_iter}.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        grad_.zero_()

                    if cfg.msk_config.buffer_type == "grad":
                        grad_ = self.momentum_grad(p_, grad_, "grad", momentum, dampening, nesterov)

                    if weight_decay != 0:
                        grad_ = grad_.add(p_, alpha=weight_decay)

                    p_.add_(grad_, alpha=-lr)

                    if cfg.msk_config.keep_min_norm:
                        match cfg.msk_config.min_norm_type:
                            case 2:
                                p_.add_(self._cal_min_norm(p_)[1])
                            case "inf":
                                sorted_mask_ = p_.sort()[0]
                                bias_ = 0.5 * (sorted_mask_[-1] + sorted_mask_[0])
                                p_.add_(-bias_)
                            case _:
                                raise NotImplementedError

                    p_.grad = None
                    del self.state[p_]["flops_grad"]
                    del self.state[p_]["tmp_grad"]
                    if "sup_soft_grad" in self.state[p_]:
                        del self.state[p_]["sup_soft_grad"]

            del self.__flops_loss
            del self.__dis_loss

    _mask_optimizer: MaskOptimizer

    def get_all_subnets(self: __DynamicResNet, **kwargs):
        if cfg.msk_config.iterative_pruning:
            if cfg.msk_config.ip_unified:
                # noinspection PyTypeChecker
                return list(self._subnet_sampler.target_subnets.values()) + [-1]
            elif cfg.msk_config.ip_pool:
                pool = self._subnet_sampler.pool
                return pool if pool else [-1]
            elif cfg.msk_config.ip_dw:
                return zip_longest(self._subnet_sampler.dw_pool, [-1], fillvalue=-1)
            else:
                return [-1]
        else:
            return self._pool

    def get_test_subnets(self: __DynamicResNet, **kwargs) -> types.GeneratorType:
        def key(_subnet: MaskedWidthProxySpaceMixIn.SubnetInfo):
            return _subnet.dis_loss.avg

        def _(meta_name, to_save=False):
            if with_meta:
                return best_subnet, meta_name, to_save
            else:
                return best_subnet

        with_meta = kwargs.get("with_meta", False)

        if cfg.msk_config.iterative_pruning:
            if cfg.msk_config.ip_unified:
                for flops, best_subnet in self._subnet_sampler.target_subnets.items():
                    yield _(f"Hard_{flops}", True)

            if cfg.msk_config.ip_pool:
                pool = self._subnet_sampler.pool
                if with_meta:
                    best_subnet = pool[np.random.randint(0, len(pool))] if pool else -1
                    yield _("Hard", True)
                else:
                    yield from (pool if pool else [-1])
                return

            if cfg.msk_config.ip_dw:
                pool = self._subnet_sampler.dw_pool
                if with_meta:
                    sorted_pool = sorted(pool.items(), key=lambda x: x[1].dis_metric.ema)
                    selected = np.random.randint(0, min(3, len(pool)))
                    best_subnet = (sorted_pool[selected][0], -1)
                    yield _("Hard", True)
                else:
                    yield from zip_longest(self._subnet_sampler.dw_pool, [-1], fillvalue=-1)
                return

            best_subnet = -1
            yield _("Hard", True)
            return

        # TODO: Refine the logic of selecting the best subnet if there are multiple targets.
        if cfg.msk_config.multi_target:
            tar_idx = cfg.msk_config.multi_target.index(cfg.msk_config.target_flops)
            best_subnet = self._pool[tar_idx]
        else:
            best_subnet = min(self._pool, key=key)

        self._req_soft_forward = True
        yield _("Soft")
        self._req_soft_forward = False
        yield _("Hard", True)
        del self._req_soft_forward

    if cfg.msk_config.iterative_pruning:

        class CircleSampler:
            if cfg.msk_config.ip_dw:
                if cfg.msk_config.ip_dw_indie_ch and cfg.msk_config.ip_mcts:
                    __max_output: Tensor | None = None
                    __input_teacher: Tensor | None = None

                    @dataclass
                    class MCTSNode:
                        reward: float = 0.0
                        count: int = 0
                        n: int = 0

                    def set_outer_vars(self, max_output: Tensor, input_teacher: Tensor):
                        self.__max_output = max_output
                        self.__input_teacher = input_teacher

                @dataclass
                class PoolAttr:
                    arch: np.ndarray
                    loop_num: int
                    circle_idx: int
                    random_seq: np.ndarray | None
                    base_flops_ratio: float
                    flops_satisfied: bool
                    dis_metric: AverageMeter = field(
                        default_factory=partial(
                            AverageMeter,
                            ema_coef=None if cfg.msk_config.ip_pool_ema == -1 else cfg.msk_config.ip_pool_ema,
                        )
                    )

                    if cfg.msk_config.ip_dw_indie_ch:
                        channel_nums: list[Tensor] = field(default_factory=list)
                        ch_divisor: list[float] = field(default_factory=list)
                        flops_reduce: list[float] = field(default_factory=list)
                        wd_mask: np.ndarray | None = None

                        if cfg.msk_config.ip_mcts:
                            mcts_cache: list["MaskedWidthProxySpaceMixIn.CircleSampler.MCTSNode"] | None = None

            def __init__(self, ctx):
                self.ctx: Union[MaskedWidthProxySpaceMixIn, DynamicResNetBase] = ctx
                self.random_seq: Optional[np.array] = None
                self.circle_idx: int = 0
                self.loop_num: int = 0
                self.metric_manager: dict[str, list[AverageMeter]] = {}

                if cfg.msk_config.ip_tb:
                    self.metric_manager["default"] = [AverageMeter() for _ in range(len(self.ctx._dep_groups))]

                if cfg.msk_config.ip_unified:
                    assert cfg.msk_config.target_flops == cfg.msk_config.multi_target[0]
                    self.targets: list[int] = cfg.msk_config.multi_target[1:]
                    self.target_subnets: dict[int, np.ndarray] = {}

                if cfg.msk_config.ip_bal:
                    self.cache_metrics: Optional[dict[int, float]] = None

                if cfg.msk_config.ip_dy_div:
                    # [increase_freq, decrease_freq]
                    self.freq_counter: list[list[int]] = [[0, 0] for _ in range(len(self.ctx._dep_groups))]

                if cfg.msk_config.ip_rewind:
                    assert not cfg.msk_config.ip_reverse
                    rewind_freq = cfg.msk_config.ip_rewind_config[0]
                    self.rewind_counter: int = 0
                    # key: group index, value: list of (the corresponding divisor, flops reduction, current metric)
                    self.rewind_cache: dict[int, list[tuple[int, float, float]]] = {
                        i: [] for i in range(len(self.ctx._dep_groups))
                    }
                    self.recent_cache: SliceableDeque[int] = SliceableDeque(maxlen=rewind_freq)
                    self.metric_manager.update(
                        {"rewind": [AverageMeter(avg_window=rewind_freq) for _ in range(len(self.ctx._dep_groups))]}
                    ) if cfg.msk_config.ip_tb else None

                if cfg.msk_config.ip_grad_div:
                    self.metric_manager.update({"grad": [AverageMeter() for _ in range(len(self.ctx._dep_groups))]})

                if cfg.msk_config.ip_pool:
                    self.pool: list[Tensor] = []
                    self.pool_buffer: list[Tensor] = []
                    self.metric_manager.update({"pool": []})

                if cfg.msk_config.ip_dw:
                    self.dw_pool: dict[
                        tuple[int, ...],
                        MaskedWidthProxySpaceMixIn.CircleSampler.PoolAttr,
                    ] = {}
                    self.all_flops: dict[tuple[int, int], float] = {}

                if cfg.msk_config.ip_dw_all:
                    self.latest_shrink_epoch: int | None = None

                if cfg.msk_config.ip_all_rand_sample:
                    self.iter_counter: int = 0

            def __getstate__(self):
                rt = [
                    self.random_seq,
                    self.circle_idx,
                    self.loop_num,
                    self.metric_manager,
                ]

                if cfg.msk_config.ip_unified:
                    rt.extend([self.targets, self.target_subnets])

                if cfg.msk_config.ip_bal:
                    rt.append(self.cache_metrics)

                if cfg.msk_config.ip_dy_div:
                    rt.append(self.freq_counter)

                if cfg.msk_config.ip_rewind:
                    rt.extend([self.rewind_counter, self.rewind_cache, self.recent_cache])

                if cfg.msk_config.ip_pool:
                    rt.append(self.pool)

                if cfg.msk_config.ip_dw:
                    rt.extend([self.dw_pool, self.all_flops])

                if cfg.msk_config.ip_dw_all:
                    rt.append(self.latest_shrink_epoch)

                if cfg.msk_config.ip_all_rand_sample:
                    rt.append(self.iter_counter)

                return tuple(rt)

            def __setstate__(self, state):
                (
                    self.random_seq,
                    self.circle_idx,
                    self.loop_num,
                    self.metric_manager,
                    *rest,
                ) = state

                if cfg.msk_config.ip_unified:
                    self.targets, self.target_subnets, *rest = rest

                if cfg.msk_config.ip_bal:
                    self.cache_metrics, *rest = rest

                if cfg.msk_config.ip_dy_div:
                    self.freq_counter, *rest = rest

                if cfg.msk_config.ip_rewind:
                    self.rewind_counter, self.rewind_cache, self.recent_cache, *rest = rest

                if cfg.msk_config.ip_pool:
                    self.pool, *rest = rest

                if cfg.msk_config.ip_dw:
                    self.dw_pool, self.all_flops, *rest = rest

                if cfg.msk_config.ip_dw_all:
                    self.latest_shrink_epoch, *rest = rest

                if cfg.msk_config.ip_all_rand_sample:
                    self.iter_counter, *rest = rest

            def __iter__(self):
                return self

            def _get_ch_nums(
                self, gp: "MaskedWidthProxySpaceMixIn.DependencyGroup", gp_idx: int, dp: tuple[int, ...] | None = None
            ) -> Tensor:
                cur_depth = self.ctx._cur_depth if dp is None else dp
                if (
                    not cfg.msk_config.ip_dw
                    or not cfg.msk_config.ip_dw_indie_ch
                    or cur_depth not in self.dw_pool
                    or not self.dw_pool[cur_depth].channel_nums
                ):
                    return gp.channel_nums
                else:
                    return self.dw_pool[cur_depth].channel_nums[gp_idx]

            @property
            def avail_indices(self) -> np.ndarray:
                base_subnet = self.ctx._base_subnet
                abs_min = cfg.msk_config.ip_abs_min

                if cfg.msk_config.ip_min_ch == "auto":
                    ip_min_ch = math.floor(math.sqrt(cfg.msk_config.target_flops))
                else:
                    ip_min_ch = cfg.msk_config.ip_min_ch

                max_num = np.array([self._get_ch_nums(gp, i)[-1].item() for i, gp in enumerate(self.ctx._dep_groups)])

                min_num = max_num * ip_min_ch
                min_num = np.where(min_num > abs_min, min_num, abs_min)

                # TODO: VIT adjust
                if cfg.model == "swin" and abs_min != 0:
                    m1, m2, m3 = abs_min // 3, abs_min // 6, abs_min // 12
                    if cfg.msk_config.merge_gp:
                        min_num[0:5:2] = [m1, m2, m3]
                    else:
                        min_num[[0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]] = [m1] * 2 + [m2] * 6 + [m3] * 4

                cur_num = np.array(
                    [self._get_ch_nums(gp, i)[base_subnet[i]].item() for i, gp in enumerate(self.ctx._dep_groups)]
                )

                if cfg.msk_config.ip_reverse and self.ctx._flops_satisfied:
                    avail_indices = np.where(cur_num < max_num)[0]
                else:
                    avail_indices = np.setdiff1d(np.where(cur_num > min_num)[0], np.where(base_subnet <= 0)[0])

                return avail_indices

            @staticmethod
            def __get_new_ch_nums(md_: int, div_: int, base_: int, device: torch.device) -> tuple[Tensor, int]:
                re_ = base_ % div_
                min_ = min(base_, re_ + div_)
                new_ch_ = torch.cat(
                    [
                        torch.arange(min_, base_ + 1, div_, device=device),
                        torch.arange(base_ + div_, md_ - div_ + 1, div_, device=device)
                        if ((base_ + div_) <= (md_ - div_))
                        else torch.tensor([], device=device),
                        torch.arange(md_, md_ + 1, device=device) if base_ < md_ else torch.tensor([], device=device),
                    ]
                )
                new_idx_ = torch.where(torch.eq(new_ch_, base_))[0].item()
                return new_ch_, new_idx_

            def _alter_divisor_grad(self):
                if cfg.msk_config.ip_dw_indie_ch:
                    raise AssertionError

                base_subnet = self.ctx._base_subnet
                dep_groups: list[MaskedWidthProxySpaceMixIn.DependencyGroup] = self.ctx._dep_groups
                grad_metrics: list[Tensor] = [
                    metric.avg[dep_groups[i].channel_nums[base_subnet[i]]]
                    for i, metric in enumerate(self.metric_manager["grad"])
                ]

                print("Here")

            def _alter_divisor_pool_rand(self):
                assert not cfg.msk_config.ip_dy_div
                if cfg.msk_config.ip_dw_indie_ch:
                    raise AssertionError

                base_subnet = self.ctx._base_subnet
                dep_groups: list[MaskedWidthProxySpaceMixIn.DependencyGroup] = self.ctx._dep_groups
                device = dep_groups[0].channel_nums.device
                min_div, max_div_o, max_ratio, div_step = cfg.msk_config.ip_div_range
                altered = False

                for i, gp in enumerate(dep_groups):
                    max_div = max(
                        min(
                            max(max_div_o, int(gp.channel_nums[-1].item() * max_ratio)),
                            int(gp.channel_nums[base_subnet[i]].item() - 1),
                        ),
                        min_div,
                    )

                    new_div = np.random.randint(min_div, max_div + 1)

                    if new_div != gp.ch_divisor:
                        altered = True
                        gp.channel_nums, base_subnet[i] = self.__get_new_ch_nums(
                            gp.channel_nums[-1].item(),
                            new_div,
                            gp.channel_nums[base_subnet[i]].item(),
                            device,
                        )
                        gp.ch_divisor.copy_(new_div)

                if altered and cfg.msk_config.ip_div_record:
                    divisor_to_print = " ".join(f"{int(gp.ch_divisor)}/{int(gp.channel_nums[-1])}" for gp in dep_groups)
                    with SmarterAppend(os.path.join(cfg.log_dir, "divisor.txt")) as d_file:
                        d_file.write("iter: {}, divisor: {}\n".format(cfg.rt_iter, divisor_to_print))

            def _alter_divisor(self):
                if cfg.msk_config.ip_dw_indie_ch:
                    raise AssertionError

                base_subnet = self.ctx._base_subnet
                dep_groups: list[MaskedWidthProxySpaceMixIn.DependencyGroup] = self.ctx._dep_groups
                device = dep_groups[0].channel_nums.device
                min_div, max_div_o, max_ratio, div_step = cfg.msk_config.ip_div_range
                inc_thresh, dec_thresh = cfg.msk_config.ip_div_freq
                altered = False

                for i, gp in enumerate(dep_groups):
                    max_div = max(
                        min(
                            max(max_div_o, int(gp.channel_nums[-1].item() * max_ratio)),
                            int(gp.channel_nums[base_subnet[i]].item() - 1),
                        ),
                        min_div,
                    )
                    new_div = gp.ch_divisor
                    inc_freq, dec_freq = self.freq_counter[i]

                    if inc_freq >= inc_thresh:
                        assert dec_freq < dec_thresh
                        new_div = min(max_div, new_div + div_step)
                        self.freq_counter[i][0] = 0

                    if dec_freq >= dec_thresh:
                        assert inc_freq < inc_thresh
                        new_div = max(min_div, new_div - div_step)
                        self.freq_counter[i][1] = 0

                    if new_div != gp.ch_divisor:
                        altered = True
                        gp.channel_nums, base_subnet[i] = self.__get_new_ch_nums(
                            gp.channel_nums[-1].item(),
                            new_div,
                            gp.channel_nums[base_subnet[i]].item(),
                            device,
                        )
                        gp.ch_divisor.copy_(new_div)

                if altered and cfg.msk_config.ip_div_record:
                    divisor_to_print = " ".join(f"{int(gp.ch_divisor)}/{int(gp.channel_nums[-1])}" for gp in dep_groups)
                    with SmarterAppend(os.path.join(cfg.log_dir, "divisor.txt")) as d_file:
                        d_file.write("iter: {}, divisor: {}\n".format(cfg.rt_iter, divisor_to_print))

            @staticmethod
            def __get_ch_frac(gp: DependencyGroup, b_idx: int) -> float:
                gp: MaskedWidthProxySpaceMixIn.DependencyGroup
                return (gp.channel_nums[b_idx] / gp.channel_nums[-1]).item()

            def _rewind_gp(self):
                if cfg.msk_config.ip_dw_indie_ch:
                    raise AssertionError

                base_subnet = self.ctx._base_subnet
                rewind_num, exclude_num = cfg.msk_config.ip_rewind_config[1:3]

                match cfg.msk_config.ip_rewind_exclude:
                    case "latest":
                        recent_cache = self.recent_cache[:exclude_num]
                    case "common":
                        # TakeNote: Due to the combination of `appendleft` and `most_common`, it can be assured that:
                        #  1. The mostly pruned group is the first to be excluded.
                        #  2. If pruned with the same counts, the recently pruned group is prior to be excluded.
                        recent_cache = list(
                            map(
                                operator.itemgetter(0),
                                Counter(self.recent_cache).most_common(exclude_num),
                            )
                        )
                    case "dis":
                        assert cfg.msk_config.ip_tb
                        recent_cache = next(
                            zip(
                                *sorted(
                                    [(idx, metric.avg) for idx, metric in enumerate(self.metric_manager["default"])],
                                    key=operator.itemgetter(1),
                                )[:exclude_num]
                            )
                        )
                    case "rewind":
                        assert cfg.msk_config.ip_rewind
                        recent_cache = next(
                            zip(
                                *sorted(
                                    [(idx, metric.avg) for idx, metric in enumerate(self.metric_manager["rewind"])],
                                    key=operator.itemgetter(1),
                                )[:exclude_num]
                            )
                        )
                    case "null":
                        recent_cache = []
                    case _:
                        raise NotImplementedError

                avail_gps: dict[int, MaskedWidthProxySpaceMixIn.DependencyGroup] = {
                    i: gp for i, gp in enumerate(self.ctx._dep_groups) if i not in recent_cache and self.rewind_cache[i]
                }

                for _ in range(rewind_num):
                    if not avail_gps:
                        break

                    cur_rewind_idx, cur_rewind_gp = min(
                        avail_gps.items(),
                        key=lambda i_gp: self.__get_ch_frac(i_gp[1], base_subnet[i_gp[0]]),
                    )
                    cur_cache = self.rewind_cache[cur_rewind_idx]
                    assert cur_cache

                    cur_div, cur_fp_rec, cur_metric = cur_cache.pop()
                    if not cur_cache:
                        del avail_gps[cur_rewind_idx]

                    cur_rewind_gp.channel_nums, base_subnet[cur_rewind_idx] = self.__get_new_ch_nums(
                        cur_rewind_gp.channel_nums[-1].item(),
                        cur_rewind_gp.ch_divisor,
                        cur_rewind_gp.channel_nums[base_subnet[cur_rewind_idx]].item() + cur_div,
                        cur_rewind_gp.channel_nums.device,
                    )
                    self.ctx._base_flops_ratio += cur_fp_rec

            @staticmethod
            def _single_layer_grad_weight_sum(layer: nn.Module, l1_order: Tensor) -> Tensor:
                # noinspection PyUnresolvedReferences
                match layer:
                    case nn.Conv2d():
                        return (layer.weight.data * layer.weight.grad).sum(dim=(1, 2, 3))[l1_order]
                    case nn.Linear():
                        return (layer.weight.data * layer.weight.grad).sum(dim=1)[l1_order]
                    case nn.Module(qkv=nn.Linear(), dim=dim, num_heads=num_heads) if (
                        cfg.msk_config.attn_wd_rep == "head_dim"
                    ):
                        # WindowAttention for Swin
                        temp = (layer.qkv.weight.data * layer.qkv.weight.grad).sum(dim=1)[l1_order]
                        # Sum again, "(c h d) -> d (c h)", only keep the `d` dimension.
                        return rearrange(
                            temp,
                            "(c h d) -> d (c h)",
                            c=3,
                            h=num_heads,
                            d=dim // num_heads,
                        ).sum(dim=1)
                    case _:
                        raise NotImplementedError

            @torch.no_grad()
            def accu_grad_metrics(self, tmp):
                dep_groups: list[MaskedWidthProxySpaceMixIn.DependencyGroup] = self.ctx._dep_groups

                # The lower, the better.
                grad_metrics = [
                    -recumsum(
                        sum(
                            map(
                                partial(
                                    self._single_layer_grad_weight_sum,
                                    l1_order=gp.l1_order,
                                ),
                                gp.layers.values(),
                            )
                        )
                    )
                    for gp in dep_groups
                ]

                # real: 3, 2, 4, 3, 1, 6, 52, 25, 22
                # es: 3, 2, null, 1, 1, 12, 21, null, null
                for gm, mmg in zip(grad_metrics, self.metric_manager["grad"]):
                    mmg.update(gm)

            @torch.no_grad()
            def _mcts_metrics(self):
                if self.__max_output is None or self.__input_teacher is None:
                    raise AssertionError

                base_subnet = self.ctx._base_subnet
                cur_depth = self.ctx._cur_depth
                base_flops_ratio = self.ctx._base_flops_ratio
                max_output, input_teacher = self.__max_output, self.__input_teacher
                dp_attr = self.dw_pool[cur_depth]
                flops_reduce = dp_attr.flops_reduce
                eps = 1e-12

                if dp_attr.mcts_cache is None:
                    dp_attr.mcts_cache = [self.MCTSNode() for _ in range(len(base_subnet))]
                mcts_cache = dp_attr.mcts_cache
                action_cache = []
                base_offset = base_subnet / base_subnet.sum()

                for _ in range(cfg.msk_config.ip_mcts_sim):
                    cur_flops_ratio = base_flops_ratio
                    cur_subnet = base_subnet.copy()
                    actions = [0 for _ in range(len(base_subnet))]
                    all_rewards = np.array([mc.reward for mc in mcts_cache])
                    normed_rewards = 1 - (all_rewards / (sum(all_rewards) + eps))
                    all_ns = np.array([mc.n for mc in mcts_cache])
                    uct = (
                        normed_rewards
                        + cfg.msk_config.ip_mcts_count_coef * np.sqrt(2 * np.log(max(1, all_ns.sum())) / (all_ns + eps))
                    ) * base_offset

                    while (cur_flops_ratio - cfg.msk_config.target_flops) > cfg.msk_config.flops_relax:
                        self.ctx._base_subnet = cur_subnet
                        sample_space = self.avail_indices
                        uct_cut = uct[sample_space]
                        sample_prob = uct_cut / uct_cut.sum()
                        action = broadcast_one_object(np.random.choice(sample_space, p=sample_prob), 0, cfg.is_ddp)
                        cur_subnet[action] -= 1
                        actions[action] += 1
                        cur_flops_ratio -= flops_reduce[action]

                    if not any(actions):
                        warnings.warn(
                            "This is a rare case that the MCTS simulation does not work as expected, "
                            "please check the implementation.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break

                    if actions in action_cache:
                        continue

                    self.ctx.set_running_subnet(proxy=cur_subnet)
                    out = cfg.rt_model(input_teacher)
                    self.ctx._reset_accumulator()
                    self.ctx.reset_running_subnet()
                    self.ctx._cur_depth = cur_depth

                    if not cfg.st_auto_t:
                        # noinspection PyUnboundLocalVariable
                        sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                            F.log_softmax(out, dim=1), F.softmax(max_output, dim=1)
                        )
                    else:
                        normalized_shape = (out.size()[1],)
                        max_output_auto_t = F.layer_norm(max_output, normalized_shape, None, None, 1e-7)
                        out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                        sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                            F.log_softmax(out_auto_t, dim=1),
                            F.softmax(max_output_auto_t, dim=1),
                        )

                    action_cache.append(actions)
                    action_nums = sum(actions)
                    # noinspection PyTypeChecker
                    ori_reward = all_reduce_avg(-sup_loss, cfg.is_ddp).item()
                    action_rewards = [ori_reward * _a / action_nums for _a in actions]

                    for node, ac, acr in zip(mcts_cache, actions, action_rewards):
                        if ac <= 0:
                            continue

                        node.count += 1
                        node.n += ac
                        node.reward += 1 / (node.count + eps) * (acr - node.reward)

                self.ctx._base_subnet = base_subnet
                return {idx: -dp_attr.mcts_cache[idx].reward for idx in self.avail_indices}

            def prune(self):
                base_subnet = self.ctx._base_subnet
                dep_groups: list[MaskedWidthProxySpaceMixIn.DependencyGroup] = self.ctx._dep_groups
                avail_indices = self.avail_indices

                match cfg.msk_config.ip_metric:
                    case "dis":
                        if cfg.msk_config.ip_dw:
                            # TODO: `avg` or `ema`?
                            p_metrics = (
                                (
                                    idx,
                                    getattr(
                                        dep_groups[idx].prune_metric["dis"][self.ctx._cur_depth],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                        else:
                            p_metrics = (
                                (
                                    idx,
                                    getattr(
                                        dep_groups[idx].prune_metric["dis"],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                    case "gt":
                        if cfg.msk_config.ip_dw:
                            p_metrics = (
                                (
                                    idx,
                                    getattr(
                                        dep_groups[idx].prune_metric["gt"][self.ctx._cur_depth],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                        else:
                            p_metrics = (
                                (
                                    idx,
                                    getattr(
                                        dep_groups[idx].prune_metric["gt"],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                    case "acc":
                        if cfg.msk_config.ip_dw:
                            p_metrics = (
                                (
                                    idx,
                                    getattr(
                                        dep_groups[idx].prune_metric["acc"][self.ctx._cur_depth],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                        else:
                            p_metrics = (
                                (
                                    idx,
                                    -getattr(
                                        dep_groups[idx].prune_metric["acc"],
                                        "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                    ),
                                )
                                for idx in avail_indices
                            )
                    case "dis_gt":
                        if cfg.msk_config.ip_dw:
                            dis_metrics = [
                                getattr(
                                    dep_groups[idx].prune_metric["dis"][self.ctx._cur_depth],
                                    "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                )
                                for idx in avail_indices
                            ]
                            gt_metrics = [
                                getattr(
                                    dep_groups[idx].prune_metric["gt"][self.ctx._cur_depth],
                                    "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                )
                                for idx in avail_indices
                            ]
                        else:
                            dis_metrics = [
                                getattr(
                                    dep_groups[idx].prune_metric["dis"],
                                    "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                )
                                for idx in avail_indices
                            ]
                            gt_metrics = [
                                getattr(
                                    dep_groups[idx].prune_metric["gt"],
                                    "avg" if not cfg.msk_config.ip_force_ema else "ema",
                                )
                                for idx in avail_indices
                            ]
                        dis_rank = scipy.stats.rankdata(dis_metrics, method="ordinal", nan_policy="raise")
                        gt_rank = scipy.stats.rankdata(gt_metrics, method="ordinal", nan_policy="raise")
                        avg_rank = (dis_rank + gt_rank) / 2
                        p_metrics = zip(avail_indices, avg_rank)
                    case _:
                        raise NotImplementedError

                if cfg.msk_config.ip_tb:
                    p_metrics = list(p_metrics)
                    default_mm = self.metric_manager["default"]
                    [default_mm[idx].update(metric) for idx, metric in p_metrics]
                    cfg.rt_tb_logger.add_scalars(
                        "prune_metric",
                        {f"gp_{idx}": metric for idx, metric in p_metrics},
                        cfg.rt_iter,
                    )
                    cfg.rt_tb_logger.add_scalars(
                        "prune_metric_avg",
                        {f"gp_{idx}": metric.avg for idx, metric in enumerate(default_mm)},
                        cfg.rt_iter,
                    )

                if cfg.msk_config.ip_bal:
                    # TODO: add compatibility for `ip_reverse`.
                    assert not cfg.msk_config.ip_reverse
                    p_metrics = dict(p_metrics)
                    cache_metrics = self.cache_metrics
                    ip_bal_ema = cfg.msk_config.ip_bal_ema

                    if cache_metrics is None:
                        self.cache_metrics = p_metrics
                        [metric.reset() for gp in dep_groups for metric in gp.prune_metric.values()]
                        return
                    else:
                        self.cache_metrics = {
                            idx: (1 - ip_bal_ema) * cache_metrics[idx] + ip_bal_ema * metric
                            for idx, metric in p_metrics.items()
                        }

                    # `cache_metrics` always contains the indices in `p_metrics` if `ip_reverse` is disabled.
                    p_metrics = {idx: metric / cache_metrics[idx] for idx, metric in p_metrics.items()}
                    p_metrics = p_metrics.items()

                if cfg.msk_config.ip_dw and cfg.msk_config.ip_dw_indie_ch and cfg.msk_config.ip_mcts:
                    gb_metrics = self._mcts_metrics()
                    p_metrics = dict(p_metrics)
                    eps = 1e-12

                    if len(gb_metrics) != len(p_metrics):
                        raise AssertionError

                    if cfg.msk_config.ip_mcts_merge == "rank":
                        gb_rank = scipy.stats.rankdata(list(gb_metrics.values()), method="ordinal", nan_policy="raise")
                        p_rank = scipy.stats.rankdata(list(p_metrics.values()), method="ordinal", nan_policy="raise")
                        avg_rank = gb_rank * cfg.msk_config.ip_mcts_merge_factor + p_rank * (
                            1 - cfg.msk_config.ip_mcts_merge_factor
                        )
                        p_metrics = zip(avail_indices, avg_rank)
                        gp_to_prune, min_metric = min(p_metrics, key=operator.itemgetter(1))

                    elif cfg.msk_config.ip_mcts_merge == "prob":
                        mcts_cache = self.dw_pool[self.ctx._cur_depth].mcts_cache
                        base_offset = base_subnet / base_subnet.sum()

                        all_rewards = np.array([mcts_cache[idx].reward for idx in avail_indices])
                        normed_rewards = 1 - (all_rewards / (sum(all_rewards) + eps))

                        p_all_rewards = np.array([p_metrics[idx] for idx in avail_indices])
                        p_normed_rewards = 1 - (p_all_rewards / (sum(p_all_rewards) + eps))

                        all_ns = np.array([mcts_cache[idx].n for idx in avail_indices])
                        uct = (
                            normed_rewards * cfg.msk_config.ip_mcts_merge_factor
                            + p_normed_rewards * (1 - cfg.msk_config.ip_mcts_merge_factor)
                            + cfg.msk_config.ip_mcts_count_coef
                            * np.sqrt(2 * np.log(max(1, all_ns.sum())) / (all_ns + eps))
                        ) * base_offset[avail_indices]
                        sample_prob = uct / uct.sum()

                        gp_to_prune = broadcast_one_object(
                            np.random.choice(avail_indices, p=sample_prob), 0, cfg.is_ddp
                        )
                        min_metric = p_metrics[gp_to_prune]

                    elif cfg.msk_config.ip_mcts_merge == "add":
                        # TODO: Other merging styles?
                        gb_metrics_sum = sum(gb_metrics.values()) + eps
                        normed_gb_metrics = {idx: metric / gb_metrics_sum for idx, metric in gb_metrics.items()}
                        p_metrics_sum = sum(p_metrics.values()) + eps
                        normed_p_metrics = {idx: metric / p_metrics_sum for idx, metric in p_metrics.items()}
                        p_metrics = {
                            idx: normed_gb_metrics[idx] * cfg.msk_config.ip_mcts_merge_factor
                            + normed_p_metrics[idx] * (1 - cfg.msk_config.ip_mcts_merge_factor)
                            for idx in avail_indices
                        }
                        gp_to_prune, min_metric = min(p_metrics.items(), key=operator.itemgetter(1))

                    else:
                        raise NotImplementedError
                elif cfg.msk_config.ip_rand_prune:
                    base_cut = base_subnet[avail_indices]
                    base_offset = base_cut / base_cut.sum()
                    p_metrics = dict(p_metrics)
                    np_p_metrics = np.array(list(p_metrics.values()))
                    softmax_p_metrics = np.exp(-np_p_metrics / cfg.msk_config.ip_rand_temp)
                    if os.environ.get("ISTP_AB_WDP_RAND_ALL"):
                        gp_to_prune = broadcast_one_object(np.random.choice(avail_indices), 0, cfg.is_ddp)
                    elif (softmax_p_metrics_sum := softmax_p_metrics.sum()) <= 0 or os.environ.get(
                        "ISTP_AB_WDP_RAND_BASE"
                    ):
                        if os.environ.get("ISTP_AB_WDP_RAND_DISTILL"):
                            gp_to_prune = broadcast_one_object(np.random.choice(avail_indices), 0, cfg.is_ddp)
                        else:
                            gp_to_prune = broadcast_one_object(
                                np.random.choice(avail_indices, p=base_offset), 0, cfg.is_ddp
                            )
                    elif os.environ.get("ISTP_AB_WDP_RAND_DISTILL"):
                        softmax_p_metrics /= softmax_p_metrics_sum
                        gp_to_prune = broadcast_one_object(
                            np.random.choice(avail_indices, p=softmax_p_metrics), 0, cfg.is_ddp
                        )
                    else:
                        softmax_p_metrics /= softmax_p_metrics_sum
                        base_cali = base_offset * softmax_p_metrics
                        if (base_cali_sum := base_cali.sum()) <= 0:
                            gp_to_prune = broadcast_one_object(
                                np.random.choice(avail_indices, p=base_offset), 0, cfg.is_ddp
                            )
                        else:
                            sample_prob = base_cali / base_cali_sum
                            gp_to_prune = broadcast_one_object(
                                np.random.choice(avail_indices, p=sample_prob), 0, cfg.is_ddp
                            )
                    min_metric = p_metrics[gp_to_prune]
                elif os.environ.get("ISTP_AB_WDP_MAX"):
                    gp_to_prune, min_metric = max(p_metrics, key=operator.itemgetter(1))
                elif os.environ.get("ISTP_AB_WDP_MEDIUM"):
                    p_metrics = list(p_metrics)
                    gp_to_prune, min_metric = sorted(p_metrics, key=operator.itemgetter(1))[len(p_metrics) // 2]
                else:
                    gp_to_prune, min_metric = min(p_metrics, key=operator.itemgetter(1))

                if cfg.msk_config.ip_reverse and self.ctx._flops_satisfied:
                    base_subnet[gp_to_prune] += 1
                else:
                    base_subnet[gp_to_prune] -= 1

                if cfg.msk_config.ip_rewind:
                    flops_reduction = self.ctx._base_flops_ratio - dep_groups[gp_to_prune].sub_flops
                    assert flops_reduction >= 0
                    self.rewind_counter += 1
                    self.rewind_cache[gp_to_prune].append(
                        (
                            dep_groups[gp_to_prune].ch_divisor.item(),
                            flops_reduction,
                            min_metric,
                        )
                    )
                    self.recent_cache.appendleft(gp_to_prune)

                    if cfg.msk_config.ip_tb:
                        rewind_mm = self.metric_manager["rewind"]
                        [rewind_mm[idx].update(metric) for idx, metric in p_metrics]
                        cfg.rt_tb_logger.add_scalars(
                            "rewind_metric",
                            {f"gp_{idx}": metric.avg for idx, metric in enumerate(rewind_mm)},
                            cfg.rt_iter,
                        )

                if not cfg.msk_config.ip_dw:
                    self.ctx._base_flops_ratio = dep_groups[gp_to_prune].sub_flops
                    [metric.reset() for gp in dep_groups for metric in gp.prune_metric.values()]
                else:
                    if (next_flops_ratio := dep_groups[gp_to_prune].sub_flops[self.ctx._cur_depth]) is None:
                        # Calculate the FLOPs ratio of the pruned group if it is not recorded.
                        cur_depth = self.ctx._cur_depth
                        is_transformer = (
                            "vit" in cfg.model
                            or "swin" in cfg.model
                            or "pit" in cfg.model
                            or "t2t" in cfg.model
                            or "cait" in cfg.model
                            or "bnit" in cfg.model
                        )
                        img_size = cfg.img_size if is_transformer else 224
                        with torch.no_grad():
                            device = cast(self.ctx.DependencyGroup, self.ctx._dep_groups[0]).channel_nums.device
                            rand_in = torch.randn(1, 3, img_size, img_size, device=device)
                            self.ctx.set_running_subnet(proxy=base_subnet)
                            cfg.rt_model(rand_in)
                            next_flops_ratio = self.ctx._all_soft_flops / self.all_flops[(img_size, img_size)]
                            self.ctx._reset_accumulator()
                            self.ctx.reset_running_subnet()
                            self.ctx._cur_depth = cur_depth

                    self.ctx._base_flops_ratio = next_flops_ratio
                    dep_groups[gp_to_prune].sub_flops[self.ctx._cur_depth] = None
                    if cfg.msk_config.ip_clean or not cfg.msk_config.ip_all_rand_sample:
                        [
                            metric[self.ctx._cur_depth].reset()
                            for gp in dep_groups
                            for metric in gp.prune_metric.values()
                        ]

                if cfg.msk_config.ip_rewind and self.rewind_counter == cfg.msk_config.ip_rewind_config[0]:
                    self.rewind_counter = 0
                    self._rewind_gp()

                # TakeNote: the `abs` function cannot be used here, to avoid drastic drop of the FLOPs ratio.
                if (self.ctx._base_flops_ratio - cfg.msk_config.target_flops) <= cfg.msk_config.flops_relax:
                    self.ctx._flops_satisfied = True
                else:
                    self.ctx._flops_satisfied = False

                if cfg.msk_config.ip_pool:
                    if cfg.msk_config.ip_dw_indie_ch:
                        raise AssertionError

                    pool: list[Tensor] = self.pool
                    pool_metrics: list[AverageMeter] = self.metric_manager["pool"]
                    min_count = cfg.msk_config.ip_pool_remove_min
                    pool_size, pb_size = (
                        cfg.msk_config.ip_pool_size,
                        cfg.msk_config.ip_pool_buffer,
                    )

                    # Remove redundant architectures in the pool.
                    if len(pool) > pool_size:
                        removable = sum([metric.count >= min_count for metric in pool_metrics]) == len(pool)

                        if removable:
                            remove_num = len(pool) - cfg.msk_config.ip_pool_size
                            # Remove the architectures with the largest metrics.
                            remove_indices = np.argsort([metric.avg for metric in pool_metrics])[-remove_num:]
                            remove_indices = np.sort(remove_indices)[::-1]
                            [pool.pop(idx) for idx in remove_indices]
                            [pool_metrics.pop(idx) for idx in remove_indices]

                        while self.pool_buffer and (len(pool) < pool_size + pb_size):
                            pool.append(self.pool_buffer.pop(0))
                            if (pool_ema := cfg.msk_config.ip_pool_ema) != -1:
                                pool_metrics.append(AverageMeter(ema_coef=pool_ema))
                            else:
                                pool_metrics.append(AverageMeter(avg_window=min_count))

                    # Update the pool.
                    if self.ctx._flops_satisfied:
                        new_arch = torch.stack([gp.channel_nums[si] for si, gp in zip(base_subnet, dep_groups)])

                        if len(pool) < (pool_size + pb_size):
                            pool.append(new_arch)
                            if (pool_ema := cfg.msk_config.ip_pool_ema) != -1:
                                pool_metrics.append(AverageMeter(ema_coef=pool_ema))
                            else:
                                pool_metrics.append(AverageMeter(avg_window=min_count))
                        else:
                            self.pool_buffer.append(new_arch)

                        if cfg.rt_epoch < (cfg.epochs * cfg.msk_config.ip_end_epoch):
                            self.ctx._flops_satisfied = False
                            # Enlarge the base subnet to continue pruning.
                            match cfg.msk_config.ip_pool_enlarge:
                                case "random":
                                    base_subnet[::] = [
                                        np.random.randint(base + 1, len(gp.channel_nums))
                                        if (base + 1) < len(gp.channel_nums)
                                        else (len(gp.channel_nums) - 1)
                                        for base, gp in zip(base_subnet, dep_groups)
                                    ]
                                case _:
                                    raise NotImplementedError

                            if cfg.msk_config.ip_pool_rand_div:
                                self._alter_divisor_pool_rand()

                if cfg.msk_config.ip_unified and self.targets:
                    if abs(self.ctx._base_flops_ratio - self.targets[-1]) <= cfg.msk_config.flops_relax:
                        self.target_subnets[self.targets.pop()] = base_subnet.copy()

                if cfg.msk_config.ip_dy_div:
                    min_divisor = cfg.msk_config.ip_div_range[0]
                    radical = cfg.msk_config.ip_div_radical
                    self.freq_counter[gp_to_prune][0] += 1
                    if radical:
                        self.freq_counter[gp_to_prune][1] = 0
                    for idx in range(len(dep_groups)):
                        if idx != gp_to_prune and dep_groups[idx].ch_divisor > min_divisor:
                            self.freq_counter[idx][1] += 1
                            if radical:
                                self.freq_counter[idx][0] = 0
                    self._alter_divisor()

                if cfg.msk_config.ip_grad_div:
                    self._alter_divisor_grad()

            def broadcast_metric(
                self,
                metric_dict: dict[tuple[int, ...], AverageMeter],
                cur_dp: tuple[int, ...],
                cur_metric: float,
                b_size: int,
            ):
                if not cfg.msk_config.ip_dw:
                    raise AssertionError

                max_dps = np.array(self.ctx.stage_list)
                cur_max_wds = np.array(
                    [max(1, len(self._get_ch_nums(gp, i, cur_dp)) - 1) for i, gp in enumerate(self.ctx._dep_groups)]
                )
                cur_wd_mask = self.dw_pool[cur_dp].wd_mask
                if cur_wd_mask is None:
                    cur_wd_mask = np.ones_like(cur_max_wds, dtype=bool)

                for next_dp, next_metric in metric_dict.items():
                    # Avoid empty metric, for the accurate first one.
                    if (
                        next_dp == cur_dp
                        or next_dp not in self.dw_pool
                        or self.dw_pool[next_dp].flops_satisfied
                        or next_metric.count == 0
                    ):
                        continue

                    next_max_wds = np.array(
                        [
                            max(1, len(self._get_ch_nums(gp, i, next_dp)) - 1)
                            for i, gp in enumerate(self.ctx._dep_groups)
                        ]
                    )

                    next_wd_mask = self.dw_pool[next_dp].wd_mask
                    if next_wd_mask is None:
                        next_wd_mask = np.ones_like(next_max_wds, dtype=bool)
                    wd_mask = cur_wd_mask & next_wd_mask

                    next_norm_dp_wd = np.concatenate([max_dps, next_max_wds[wd_mask]])
                    next_dp_wd = np.concatenate([np.array(next_dp), self.dw_pool[next_dp].arch[wd_mask]])

                    cur_norm_dp_wd = np.concatenate([max_dps, cur_max_wds[wd_mask]])
                    cur_dp_wd = np.concatenate([np.array(cur_dp), self.dw_pool[cur_dp].arch[wd_mask]])

                    match os.environ.get("ISTP_AB_BM_TYPE"):
                        case "l1":
                            arch_diff = np.linalg.norm(
                                np.abs(cur_dp_wd / cur_norm_dp_wd - next_dp_wd / next_norm_dp_wd), ord=1
                            ) / len(cur_norm_dp_wd)
                        case "w":
                            # noinspection PyTypeChecker
                            arch_diff = (
                                scipy.stats.wasserstein_distance(
                                    np.arange(len(cur_dp_wd)),
                                    np.arange(len(next_dp_wd)),
                                    cur_dp_wd / cur_norm_dp_wd,
                                    next_dp_wd / next_norm_dp_wd,
                                )
                                / len(cur_norm_dp_wd)
                                * 10
                            )
                        case "js":
                            arch_diff = (
                                scipy.spatial.distance.jensenshannon(
                                    cur_dp_wd / cur_norm_dp_wd, next_dp_wd / next_norm_dp_wd
                                )
                                / len(cur_norm_dp_wd)
                                * 10
                            )
                        case _:
                            arch_diff = np.linalg.norm(
                                np.abs(cur_dp_wd / cur_norm_dp_wd - next_dp_wd / next_norm_dp_wd), ord=2
                            ) ** 2 / len(cur_norm_dp_wd)

                    if cfg.msk_config.ip_broadcast_coef > 0:
                        update_coef = float(np.exp(-arch_diff / cfg.msk_config.ip_broadcast_coef))
                    else:
                        update_coef = 0

                    # TODO: `avg` or `ema`?
                    if cfg.msk_config.ip_pool_ema == -1:
                        raise AssertionError

                    update_var = next_metric.ema * (1 - update_coef) + cur_metric * update_coef
                    next_metric.update(update_var, b_size)
                    next_metric.sync_metrics()

            @torch.no_grad()
            def _recali_flops(self):
                # TODO: Add support for extreme merge.
                if cfg.msk_config.extreme_merge:
                    return

                # TODO: Add support for non-DW.
                if not cfg.msk_config.ip_dw:
                    return

                global GB_MAKE_DIVISIBLE_TENSOR_DIVISOR
                GB_MAKE_DIVISIBLE_TENSOR_DIVISOR = 1
                cur_depth = self.ctx._cur_depth
                dep_groups = cast(list[MaskedWidthProxySpaceMixIn.DependencyGroup], self.ctx._dep_groups)

                is_transformer = (
                    "vit" in cfg.model
                    or "swin" in cfg.model
                    or "pit" in cfg.model
                    or "t2t" in cfg.model
                    or "cait" in cfg.model
                    or "bnit" in cfg.model
                )
                img_size = cfg.img_size if is_transformer else 224
                device = dep_groups[0].channel_nums.device
                rand_in = torch.randn(1, 3, img_size, img_size, device=device)

                for dp, dp_attr in self.dw_pool.items():
                    if dp_attr.channel_nums:
                        if len(dp_attr.channel_nums) != len(dp_attr.ch_divisor):
                            raise AssertionError
                    diff_list = []
                    base_subnet = dp_attr.arch
                    dp_attr.flops_reduce.clear()
                    ch_nums = torch.tensor([dp_attr.channel_nums[i][base_subnet[i]] for i, gp in enumerate(dep_groups)])

                    self.ctx._cur_depth = dp
                    self.ctx.set_running_subnet(ch_nums)
                    cfg.rt_model(rand_in)
                    cur_total_flops = self.ctx._all_soft_flops.item()
                    self.ctx._reset_accumulator()

                    for cur_idx in range(len(ch_nums)):
                        cur_subnet = ch_nums.clone()
                        if cur_subnet[cur_idx] == 0:
                            diff_list.append(0)
                            continue
                        cur_subnet[cur_idx] -= 1
                        self.ctx.set_running_subnet(cur_subnet)
                        cfg.rt_model(rand_in)
                        diff_list.append(max(0, cur_total_flops - self.ctx._all_soft_flops.item()))
                        dp_attr.flops_reduce.append(diff_list[-1] / self.all_flops[(img_size, img_size)])
                        self.ctx._reset_accumulator()

                    max_idx = np.argmax(diff_list)
                    max_diff = diff_list[max_idx]

                    def _handle_zero(_m_d: float, _d: float, _i: int):
                        try:
                            return _m_d / _d
                        except ZeroDivisionError:
                            return dep_groups[_i].channel_nums[-1].item()

                    ori_divisor = [1.0 for _ in range(len(dep_groups))]
                    scaled_diff = [
                        _handle_zero(max_diff, diff, i) * cfg.msk_config.ip_recalc_scale
                        for i, diff in enumerate(diff_list)
                    ]

                    def _get_new_ch_nums(md_: int, base_: int, div_: int) -> tuple[Tensor, int, int]:
                        div_ = max(min(div_ * cfg.msk_config.ip_recalc_scale, base_ - 1), 1)
                        min2_ = min_ = max(min(int(md_ * cfg.msk_config.min_width), base_), 1)
                        while ((base_ - min_) % div_) != 0:
                            if min_ == 1:
                                break
                            min_ -= 1
                        while min2_ < min_:
                            min2_ += div_
                        min_ = max(min(min2_, base_), 1)
                        new_ch_ = torch.cat(
                            [
                                torch.arange(min_, base_ + 1, div_, device=device),
                                torch.arange(base_ + div_, md_, div_, device=device)
                                if ((base_ + div_) <= (md_ - 1))
                                else torch.tensor([], device=device, dtype=torch.int),
                                torch.arange(md_, md_ + 1, device=device)
                                if base_ < md_
                                else torch.tensor([], device=device, dtype=torch.int),
                            ]
                        )
                        new_idx_ = torch.where(torch.eq(new_ch_, base_))[0].item()
                        return new_ch_, new_idx_, int(div_)

                    dp_attr.channel_nums.clear()
                    dp_attr.ch_divisor.clear()
                    for i, gp in enumerate(dep_groups):
                        new_ch_nums, dp_attr.arch[i], new_div = _get_new_ch_nums(
                            gp.channel_nums[-1].item(), ch_nums[i].item(), round(scaled_diff[i])
                        )
                        dp_attr.channel_nums.append(new_ch_nums)
                        dp_attr.ch_divisor.append(new_div)

                    dp_attr.flops_reduce = [
                        (_fr * _post_d / _pre_d)
                        for _fr, _pre_d, _post_d in zip(dp_attr.flops_reduce, ori_divisor, dp_attr.ch_divisor)
                    ]

                    if cfg.msk_config.ip_recali_flops_verbose:
                        cfg.rt_logger.info(f"The recalibrated divisor is: {dp_attr.ch_divisor}.")

                    self.ctx.reset_running_subnet()

                self.ctx._cur_depth = cur_depth
                GB_MAKE_DIVISIBLE_TENSOR_DIVISOR = None

            def _all_rand_sample_next(self):
                self.iter_counter += 1

                if (
                    self.iter_counter % cfg.msk_config.ip_prune_freq == 0
                    and not all(p_m.flops_satisfied for p_m in self.dw_pool.values())
                    # TODO: Complex metric such as "dis_gt"?
                    and all(
                        all(
                            cast(self.ctx.DependencyGroup, dg).prune_metric[cfg.msk_config.ip_metric][depth].count
                            for i, dg in enumerate(self.ctx._dep_groups)
                            if len(self._get_ch_nums(dg, i, depth)) > 1
                        )
                        for depth in self.dw_pool.keys()
                    )
                ):
                    # Unified pruning for all depths.
                    for depth, pool_attr in self.dw_pool.items():
                        if pool_attr.flops_satisfied:
                            continue
                        self.ctx._cur_depth = depth
                        self.ctx._base_subnet = pool_attr.arch
                        self.ctx._base_flops_ratio = pool_attr.base_flops_ratio
                        self.ctx._flops_satisfied = pool_attr.flops_satisfied
                        self.prune()
                        pool_attr.flops_satisfied = self.ctx._flops_satisfied
                        pool_attr.base_flops_ratio = self.ctx._base_flops_ratio

                    if cfg.msk_config.ip_recali_flops:
                        self._recali_flops()

                depth, pool_attr = list(self.dw_pool.items())[np.random.randint(0, len(self.dw_pool))]
                self.ctx._cur_depth = depth
                self.ctx._base_subnet = pool_attr.arch
                self.ctx._base_flops_ratio = pool_attr.base_flops_ratio
                self.ctx._flops_satisfied = pool_attr.flops_satisfied

                if self.ctx._flops_satisfied and (
                    not cfg.msk_config.ip_reverse or cfg.rt_epoch >= (cfg.epochs * cfg.msk_config.ip_end_epoch)
                ):
                    # TODO: Is it necessary to clean up metrics when the epoch reaches the end?
                    return depth, -1

                avail_indices = self.avail_indices
                if avail_indices.size == 0:
                    if cfg.msk_config.ip_reverse and self.ctx._flops_satisfied:
                        raise RuntimeError("All groups are fully expanded, which is not expected.")
                    else:
                        raise RuntimeError("All groups are fully pruned, which is not expected.")

                chosen_gp = int(np.random.choice(avail_indices))

                if cfg.rt_epoch < (cfg.st_warmup_epochs + cfg.msk_config.ip_warm_up):
                    chosen_gp = -1

                return depth, chosen_gp

            def _dw_pool_next(self):
                # Pool shrinkage if needed.
                # TODO: Buggy for the reduction of the pool size.
                if (
                    cfg.msk_config.ip_dw_all
                    and all(p.flops_satisfied for p in self.dw_pool.values())
                    and len(self.dw_pool) > cfg.msk_config.ip_pool_size
                ):
                    ep_left = max(1.0, cfg.msk_config.ip_end_epoch * cfg.epochs - cfg.rt_epoch)
                    sz_left = max(0, len(self.dw_pool) - cfg.msk_config.ip_pool_size)
                    ep_shrink_sz = sz_left / ep_left
                    if self.latest_shrink_epoch is None:
                        self.latest_shrink_epoch = cfg.rt_epoch
                        ep_accu = 1
                    else:
                        ep_accu = cfg.rt_epoch - self.latest_shrink_epoch

                    if (to_shrink_sz := int(ep_accu * ep_shrink_sz)) >= 1:
                        self.latest_shrink_epoch = cfg.rt_epoch

                        # Remove `to_shrink_sz` items from `self.dw_pool` based on the `PoolAttr.dis_metric.ema`,
                        # the larger ones will be removed first.
                        # TODO: `ema` or `avg`?
                        sorted_pool = sorted(
                            self.dw_pool.items(),
                            key=lambda x: x[1].dis_metric.ema,
                            reverse=True,
                        )
                        for i in range(to_shrink_sz):
                            if len(self.dw_pool) <= cfg.msk_config.ip_pool_size:
                                break
                            del self.dw_pool[sorted_pool[i][0]]

                if cfg.msk_config.ip_all_rand_sample:
                    return self._all_rand_sample_next()

                depth, pool_attr = list(self.dw_pool.items())[np.random.randint(0, len(self.dw_pool))]
                self.ctx._cur_depth = depth
                self.ctx._base_subnet = pool_attr.arch
                self.ctx._base_flops_ratio = pool_attr.base_flops_ratio
                self.ctx._flops_satisfied = pool_attr.flops_satisfied

                if pool_attr.loop_num == cfg.msk_config.ip_prune_freq:
                    self.prune()
                    pool_attr.loop_num = 0
                    pool_attr.flops_satisfied = self.ctx._flops_satisfied
                    pool_attr.base_flops_ratio = self.ctx._base_flops_ratio

                    if cfg.msk_config.ip_recali_flops:
                        self._recali_flops()

                if self.ctx._flops_satisfied and (
                    not cfg.msk_config.ip_reverse or cfg.rt_epoch >= (cfg.epochs * cfg.msk_config.ip_end_epoch)
                ):
                    # TODO: Is it necessary to clean up metrics when the epoch reaches the end?
                    return depth, -1

                if pool_attr.circle_idx == 0:
                    avail_indices = self.avail_indices
                    if avail_indices.size == 0:
                        if cfg.msk_config.ip_reverse and self.ctx._flops_satisfied:
                            raise RuntimeError("All groups are fully expanded, which is not expected.")
                        else:
                            raise RuntimeError("All groups are fully pruned, which is not expected.")

                    pool_attr.random_seq = np.random.permutation(avail_indices)

                chosen_gp = pool_attr.random_seq[pool_attr.circle_idx]
                pool_attr.circle_idx += 1

                if pool_attr.circle_idx == len(pool_attr.random_seq):
                    pool_attr.circle_idx = 0
                    pool_attr.loop_num += 1

                if cfg.rt_epoch < (cfg.st_warmup_epochs + cfg.msk_config.ip_warm_up):
                    chosen_gp = -1

                return depth, int(chosen_gp)

            def __next__(self):
                if cfg.msk_config.ip_dw:
                    return self._dw_pool_next()

                if self.loop_num == cfg.msk_config.ip_prune_freq:
                    self.prune()
                    self.loop_num = 0

                    if cfg.msk_config.ip_recali_flops:
                        self._recali_flops()

                if self.ctx._flops_satisfied and (
                    not cfg.msk_config.ip_reverse or cfg.rt_epoch >= (cfg.epochs * cfg.msk_config.ip_end_epoch)
                ):
                    # TODO: Is it necessary to clean up metrics when the epoch reaches the end?
                    return -1

                if self.circle_idx == 0:
                    avail_indices = self.avail_indices
                    if avail_indices.size == 0:
                        if cfg.msk_config.ip_reverse and self.ctx._flops_satisfied:
                            raise RuntimeError("All groups are fully expanded, which is not expected.")
                        else:
                            raise RuntimeError("All groups are fully pruned, which is not expected.")

                    self.random_seq = np.random.permutation(avail_indices)

                chosen_gp = self.random_seq[self.circle_idx]
                self.circle_idx += 1

                if self.circle_idx == len(self.random_seq):
                    self.circle_idx = 0
                    self.loop_num += 1

                if cfg.rt_epoch < (cfg.st_warmup_epochs + cfg.msk_config.ip_warm_up):
                    chosen_gp = -1

                return int(chosen_gp)

        _subnet_sampler: CircleSampler
        _base_subnet: np.ndarray
        _base_flops_ratio: float
        _flops_satisfied: bool

    def sample_subnet(self: __DynamicResNet, **kwargs) -> SubnetInfo | int:
        if cfg.msk_config.iterative_pruning:
            return next(self._subnet_sampler)
        elif len(self._pool) < cfg.msk_config.pool_size:
            raise RuntimeError("The pool is not full yet. Please generate anchors first.")
        else:
            return super().sample_subnet(**kwargs)

    def post_load_checkpoint(self: __DynamicResNet, checkpoint: dict):
        # Load an initialized pool (will not overwrite same key in the checkpoint)
        if (pl_ck := cfg.msk_config.load_init_pool) is not None:
            list(
                itertools.starmap(
                    checkpoint.setdefault,
                    torch.load(pl_ck, map_location=f"cuda:{cfg.device}").items(),
                )
            )

        if "pool" in checkpoint:
            self._pool = checkpoint["pool"]
        if "group_meta" in checkpoint:
            [
                cast(self.DependencyGroup, group).load_state_dict(state)
                for group, state in zip(self._dep_groups, checkpoint["group_meta"])
            ]
        if "mask_optimizer" in checkpoint:
            if len(self._mask_optimizer.param_groups) != len(self._pool):
                assert len(self._mask_optimizer.param_groups) == 0, "The optimizer is not empty at initialization!"
                list(
                    map(
                        self._mask_optimizer.add_param_group,
                        itertools.starmap(
                            lambda i, sn: {"idx": i, "params": sn.masks},
                            enumerate(self._pool),
                        ),
                    )
                )
            self._mask_optimizer.load_state_dict(checkpoint["mask_optimizer"])
        if cfg.msk_config.iterative_pruning:
            if "subnet_sampler" in checkpoint:
                self._subnet_sampler.__dict__.update(checkpoint["subnet_sampler"].__dict__)
            if "base_subnet" in checkpoint:
                self._base_subnet = checkpoint["base_subnet"]
            if "base_flops_ratio" in checkpoint:
                self._base_flops_ratio = checkpoint["base_flops_ratio"]
            if "flops_satisfied" in checkpoint:
                self._flops_satisfied = checkpoint["flops_satisfied"]

        return checkpoint

    def pre_save_checkpoint(self: __DynamicResNet, checkpoint: dict):
        # TODO: Unpacking without class dependency.
        if cfg.rt_epoch >= cfg.st_warmup_epochs:
            checkpoint["pool"] = self._pool
            checkpoint["group_meta"] = [group.state_dict() for group in self._dep_groups]
            checkpoint["mask_optimizer"] = self._mask_optimizer.state_dict()

            if cfg.msk_config.iterative_pruning:
                checkpoint["subnet_sampler"] = self._subnet_sampler
                checkpoint["base_subnet"] = self._base_subnet
                checkpoint["base_flops_ratio"] = self._base_flops_ratio
                checkpoint["flops_satisfied"] = self._flops_satisfied

        return checkpoint

    def _post_initialize(self: __DynamicResNet):
        super()._post_initialize()
        self.Conv2d._get_oc_rep_sigmoid = self._conv2d_get_oc_rep_sigmoid

        # Setting up the auxiliary classes.
        self.Proxy.set_max(
            tuple(next(iter(_dep_group.layers.values())).out_channels for _dep_group in self._dep_groups)
        )
        self.SubnetInfo.set_groups(self._dep_groups)
        self._mask_optimizer = self.MaskOptimizer()
        self._mask_optimizer.set_ctx(self)
        self.DependencyGroup.set_optimizer(self._mask_optimizer)

        self.ReLU.forward = self._relu_forward
        for n, m in self.named_modules():
            if isinstance(m, nn.ReLU) and n != "relu":
                m.__class__ = self.ReLU

        ch_divisor = cfg.msk_config.ch_divisor
        for _dep_group in self._dep_groups:
            _dep_group = cast(self.DependencyGroup, _dep_group)

            # Set name for each conv layer.
            for _mn, _m in _dep_group.layers.items():
                _m = cast(self.Conv2d, _m)
                _m._name = _mn

            # Fetch a representative conv layer for the current group, generally the first one is enough.
            module = next(iter(_dep_group.layers.values()))
            _dep_group.channel_nums, _dep_group.ch_divisor = self._divide_channel(module.out_channels, ch_divisor)
            _dep_group.l1_order = torch.arange(module.out_channels)
            _dep_group.order_initialized = torch.tensor(False)
            _dep_group.mask = {}
            _dep_group.to(torch.device(torch.cuda.current_device()))

        cfg.register_pre_batch_hook(self.__pre_batch_hook)
        cfg.register_student_pre_backward_hook(self.__student_pre_backward_hook)
        cfg.register_post_student_hook(self.__post_student_hook)

        if cfg.msk_config.iterative_pruning:
            divisor_to_print = " ".join(
                f"{int(gp.ch_divisor)}/{int(gp.channel_nums[-1])}"
                for gp in cast(list[self.DependencyGroup], self._dep_groups)
            )
            cfg.rt_logger.info(f"The divisor/channel of groups: <{divisor_to_print}>")

            self._subnet_sampler = self.CircleSampler(self)
            self._base_subnet = np.array(
                [
                    np.ceil((len(cast(self.DependencyGroup, gp).channel_nums) - 1) * cfg.msk_config.ip_start_ratio)
                    for gp in self._dep_groups
                ],
                dtype=np.int64,
            )
            self._base_flops_ratio = 1.0
            self._flops_satisfied = False

    if cfg.msk_config.ip_dw:
        _cur_depth: tuple[int, ...]

    def _apply_proxy(self: __DynamicResNet, subnet: SubnetInfo | int | None | np.ndarray):
        # Note that, we never clear the `_cur_subnet`.
        # As a result, the current call can always review the proxy of the previous one.
        #
        # Best subnet of Mask-ST:
        # np.array(
        #     [13, 5, 15, 5, 13, 3, 13, 11, 15, 16, 3, 3, 3, 3, 13, 2, 1, 1, 2, 15, 13, 14, 12, 12, 13, 14, 13]
        # )
        if isinstance(subnet, tuple):
            assert cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw
            self._cur_depth, subnet = subnet
            cur_attr = self._subnet_sampler.dw_pool[self._cur_depth]
            self._base_subnet = cur_attr.arch
            self._flops_satisfied = cur_attr.flops_satisfied
        if isinstance(subnet, int):
            assert cfg.msk_config.iterative_pruning

            if subnet == -1:
                cur_subnet = self._base_subnet
            elif cfg.msk_config.ip_reverse and self._flops_satisfied:
                cur_subnet = [s + 1 if i == subnet else s for i, s in enumerate(self._base_subnet)]
            else:
                cur_subnet = [s - 1 if i == subnet else s for i, s in enumerate(self._base_subnet)]

            for _i, _dep_group in enumerate(self._dep_groups):
                _dep_group = cast(self.DependencyGroup, _dep_group)
                if cfg.msk_config.extreme_merge:
                    cur_frac = (
                        self._subnet_sampler._get_ch_nums(_dep_group, _i)
                        if cfg.msk_config.iterative_pruning
                        else _dep_group.channel_nums
                    )[cur_subnet[_i]]
                    _sub_dep_groups = cast(list[self.DependencyGroup], _dep_group.sub_gps)
                    for _sub_dep_group in _sub_dep_groups:
                        _sub_dep_group.dep_info = self.DependencyInfo(
                            soft_out_channels=make_divisible_tensor(
                                _sub_dep_group.channel_nums[-1] * cur_frac,
                                limit=_sub_dep_group.channel_nums[-1],
                            ).float()
                        )
                else:
                    ch_nums = (
                        self._subnet_sampler._get_ch_nums(_dep_group, _i)
                        if cfg.msk_config.iterative_pruning
                        else _dep_group.channel_nums
                    )
                    _dep_group.dep_info = self.DependencyInfo(
                        soft_out_channels=make_divisible_v2(ch_nums[cur_subnet[_i]], ch_nums[-1]).float()
                    )

            self._cur_subnet = self.SubnetInfo(subnet, self.Proxy(), cfg.msk_config.history_num)
            return
        elif isinstance(subnet, np.ndarray):
            assert cfg.msk_config.iterative_pruning

            for _i, _dep_group in enumerate(self._dep_groups):
                _dep_group = cast(self.DependencyGroup, _dep_group)
                if cfg.msk_config.extreme_merge:
                    cur_frac = (
                        self._subnet_sampler._get_ch_nums(_dep_group, _i)
                        if cfg.msk_config.iterative_pruning
                        else _dep_group.channel_nums
                    )[subnet[_i]]
                    _sub_dep_groups = cast(list[self.DependencyGroup], _dep_group.sub_gps)
                    for _sub_dep_group in _sub_dep_groups:
                        _sub_dep_group.dep_info = self.DependencyInfo(
                            soft_out_channels=make_divisible_tensor(
                                _sub_dep_group.channel_nums[-1] * cur_frac,
                                limit=_sub_dep_group.channel_nums[-1],
                            ).float()
                        )
                else:
                    ch_nums = (
                        self._subnet_sampler._get_ch_nums(_dep_group, _i)
                        if cfg.msk_config.iterative_pruning
                        else _dep_group.channel_nums
                    )
                    _dep_group.dep_info = self.DependencyInfo(
                        soft_out_channels=make_divisible_v2(ch_nums[subnet[_i]], ch_nums[-1]).float()
                    )

            # TODO: Better subnet index for the additional forwards, currently `-2` is used.
            self._cur_subnet = self.SubnetInfo(-2, self.Proxy(), cfg.msk_config.history_num)
            return
        elif isinstance(subnet, Tensor):
            assert cfg.msk_config.iterative_pruning

            for _i, _dep_group in enumerate(self._dep_groups):
                _dep_group = cast(self.DependencyGroup, _dep_group)
                ch_nums = (
                    self._subnet_sampler._get_ch_nums(_dep_group, _i)
                    if cfg.msk_config.iterative_pruning
                    else _dep_group.channel_nums
                )
                _dep_group.dep_info = self.DependencyInfo(
                    soft_out_channels=make_divisible_v2(subnet[_i].to(ch_nums.device), ch_nums[-1]).float()
                )

            # TODO: Better subnet index for the additional forwards, currently `-3` is used.
            self._cur_subnet = self.SubnetInfo(-3, self.Proxy(), cfg.msk_config.history_num)
            return
        elif isinstance(subnet, self.SubnetInfo):
            # For the second phase, optimizing subnet masks, and for testing.
            self._cur_subnet = subnet
        else:
            # For the first phase, finding out subnet anchors.
            self._cur_subnet = self.SubnetInfo(len(self._pool), self.Proxy(), cfg.msk_config.history_num)

        for _i, _dep_group in enumerate(self._dep_groups):
            _dep_group.dep_info = self.DependencyInfo()

    def _clear_proxy(self: __DynamicResNet):
        for _dep_group in self._dep_groups:
            _dep_group = cast(self.DependencyGroup, _dep_group)
            _dep_group.dep_info = None

            if cfg.msk_config.extreme_merge:
                for _sub_dep_group in _dep_group.sub_gps:
                    _sub_dep_group.dep_info = None

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw:
            self._cur_depth = tuple(self.stage_list)

    @property
    def _req_soft_forward(self: __DynamicResNet):
        try:
            return self.__req_soft_forward
        except AttributeError:
            return cfg.msk_config.soft_forward

    @_req_soft_forward.setter
    def _req_soft_forward(self: __DynamicResNet, value):
        self.__req_soft_forward = value

    @_req_soft_forward.deleter
    def _req_soft_forward(self: __DynamicResNet):
        try:
            del self.__req_soft_forward
        except AttributeError:
            pass

    @property
    def _req_hard_forward(self: __DynamicResNet):
        try:
            return self.__req_hard_forward
        except AttributeError:
            return not self.training or cfg.rt_mode == "reset_bn" or cfg.rt_epoch >= cfg.msk_config.hard_ft_epoch

    @_req_hard_forward.setter
    def _req_hard_forward(self: __DynamicResNet, value):
        self.__req_hard_forward = value

    @_req_hard_forward.deleter
    def _req_hard_forward(self: __DynamicResNet):
        try:
            del self.__req_hard_forward
        except AttributeError:
            pass

    __req_param_stat: bool
    # TODO: Better initialization of `_param_stat`.
    _param_stat: list[float]

    @property
    def _req_param_stat(self: __DynamicResNet):
        try:
            return self.__req_param_stat
        except AttributeError:
            return cfg.msk_config.count_param

    @_req_param_stat.setter
    def _req_param_stat(self: __DynamicResNet, value):
        self.__req_param_stat = value

    @_req_param_stat.deleter
    def _req_param_stat(self: __DynamicResNet):
        try:
            del self.__req_param_stat
        except AttributeError:
            pass

    @contextlib.contextmanager
    def _get_param_stat(self: __DynamicResNet):
        self._param_stat = [0, 0]
        self._req_param_stat = True
        yield
        del self._req_param_stat

    @property
    @lru_cache(maxsize=1)
    def _flops_threshold(self: __DynamicResNet):
        return cfg.msk_config.flops_loss_coef * (cfg.msk_config.flops_relax**2)

    if cfg.msk_config.solidify:

        @property
        @lru_cache(maxsize=1)
        def _epochs_to_solidify(self: __DynamicResNet):
            if cfg.msk_config.cosine_gap:
                s_s, s_e = cfg.msk_config.solidify_interval
                cos_peak, cos_min = s_e - s_s, s_s
                gp_num = len(self._dep_groups)
                cos_epochs = cos_peak * np.cos(np.pi / 2 * (np.arange(gp_num) / (gp_num - 1) - 1)) + cos_min
                int_epochs = np.floor(cos_epochs).astype(int)
                _, counts = np.unique(int_epochs, return_counts=True)
                assert np.all(counts == 1), "The solidify epochs are not unique!"
                return int_epochs
            else:
                return list(np.floor(np.linspace(*cfg.msk_config.solidify_interval, len(self._dep_groups))).astype(int))

        @exec_range(lambda _ep: True, ep_only_once=True)
        @torch.no_grad()
        def _solidify(self: __DynamicResNet, mask_idx: int, l_dict: dict):
            if cfg.rt_epoch not in self._epochs_to_solidify:
                return

            # Try to solidify the masks.
            cho_prob, cho_mask, cho_gp, cho_i = (
                cfg.msk_config.solidify_threshold,
                None,
                None,
                0,
            )
            gp_cls = self.DependencyGroup

            def _(i: int, gp: gp_cls):
                if gp.solidified[mask_idx]:
                    return

                nonlocal cho_prob, cho_mask, cho_gp, cho_i
                _msk = gp.mask[mask_idx]
                norm_msk = F.softmax(torch.div(_msk, cfg.msk_config.softmax_temp), dim=0)
                cur_max = torch.max(norm_msk)

                match cfg.msk_config.pick_mask:
                    case "default":
                        if cur_max > cho_prob:
                            cho_prob, cho_mask, cho_gp, cho_i = cur_max, _msk, gp, i

                    case "backward":
                        cho_prob, cho_mask, cho_gp, cho_i = cur_max, _msk, gp, i

                    case "min_gap" | "max_gap" as pick_mask:
                        nonlocal cho_gap

                        self._req_hard_forward = True
                        self._req_soft_forward = False
                        gp.solidified[mask_idx] = True

                        self.set_running_subnet(chosen_subnet)
                        out = cfg.rt_model(images)

                        if cfg.msk_config.auto_t:
                            out = F.layer_norm(out, normalized_shape, None, None, 1e-7)

                        gap = gap_metric(F.log_softmax(out, dim=1), tar_prob)
                        gap = all_reduce_avg(gap, cfg.is_ddp)
                        if (pick_mask == "min_gap" and gap < cho_gap) or (pick_mask == "max_gap" and gap > cho_gap):
                            cho_prob, cho_mask, cho_gp, cho_i, cho_gap = (
                                cur_max,
                                _msk,
                                gp,
                                i,
                                gap,
                            )

                        gp.solidified[mask_idx] = False
                        del self._req_soft_forward
                        del self._req_hard_forward

                    case _:
                        raise NotImplementedError(f"Unsupported mask picking strategy: {cfg.msk_config.pick_mask}!")

            cho_gap = torch.inf if cfg.msk_config.pick_mask == "min_gap" else 0
            if cfg.msk_config.pick_mask in ("min_gap", "max_gap"):
                chosen_subnet = l_dict["chosen_subnet"]
                images = l_dict["images"]
                gap_metric = torch.nn.KLDivLoss(reduction="batchmean")

                is_transformer = (
                    "vit" in cfg.model
                    or "swin" in cfg.model
                    or "pit" in cfg.model
                    or "t2t" in cfg.model
                    or "cait" in cfg.model
                    or "bnit" in cfg.model
                )
                resolution = max(cfg.resos) if not is_transformer else cfg.img_size
                images = F.interpolate(
                    images,
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=True,
                )

                self._req_hard_forward = False
                self._req_soft_forward = True

                self.set_running_subnet(chosen_subnet)
                soft_out = cfg.rt_model(images)

                normalized_shape = (soft_out.size()[1],)
                if not cfg.msk_config.auto_t:
                    tar_prob = F.softmax(soft_out, dim=1)
                else:
                    tar_prob = F.softmax(
                        F.layer_norm(soft_out, normalized_shape, None, None, 1e-7),
                        dim=1,
                    )

                del self._req_soft_forward
                del self._req_hard_forward

            list(itertools.starmap(_, enumerate(self._dep_groups)))

            if cho_mask is not None:
                assert cho_gp is not None
                if not cfg.msk_config.solidify_grad:
                    cho_mask.requires_grad = False
                cast(gp_cls, cho_gp).solidified[mask_idx] = True
                cfg.rt_logger.info(
                    f"Solidify the {cho_i}-th mask of the {mask_idx}-th mask set "
                    f"with the maximum probability: {cho_prob:.4f}." + f" (Gap: {cho_gap:.4f})"
                    if cfg.msk_config.pick_mask in ("min_gap", "max_gap")
                    else ""
                )

    def _reset_accumulator(self):
        self._all_soft_flops = 0.0
        self._all_flops = 0.0
        self._all_norm = 0.0
        self._all_entropy = 0.0
        self._param_stat = [0, 0]

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.ReLU.forward)
    def _relu_forward(
        ctx: __DynamicResNet,
        self: ReLU,
        input: TensorWithMeta,
    ) -> TensorWithMeta:
        in_meta: ctx.ForwardMeta
        input, in_meta = input.tensor, input.meta_data

        out = F.relu(input, inplace=self.inplace)
        if in_meta.mask is not None:
            out = ctx._gw.watch(out * in_meta.mask, f"{in_meta.conv_name}->relu<forward>: output")

        return TensorWithMeta(out, in_meta)

    @MultiBindMethod
    @DynamicResNetBase.student_only(nn.Conv2d.forward)
    def _conv2d_forward(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: TensorWithMeta,
    ) -> TensorWithMeta:
        in_meta: ctx.ForwardMeta
        input, in_meta = input.tensor, input.meta_data
        group: ctx.DependencyGroup = self._group
        dep_info: Optional[ctx.DependencyInfo] = group.dep_info

        req_soft_fd = ctx._req_soft_forward or (cfg.msk_config.solidify and not group.solidified[ctx._cur_subnet.idx])
        use_relu_mask = cfg.msk_config.relu_mask or (
            cfg.msk_config.solidify and cfg.msk_config.solidify_relu and group.solidified[ctx._cur_subnet.idx]
        )

        if dep_info.out_indices is None:
            if cfg.msk_config.iterative_pruning:
                soft_out_channels = int(dep_info.soft_out_channels.item())
                ctx._cur_subnet.proxy[self._name] = soft_out_channels
                ctx._cur_subnet.proxy._cur[ctx._dep_groups.index(group)] = soft_out_channels
                dep_info.diff_mask = torch.zeros(self.out_channels, device=input.device, dtype=torch.float32)
                dep_info.diff_mask[:soft_out_channels] = 1.0
                dep_info.out_indices = self._get_out_indices(input, in_meta, dep_info, dep_info.diff_mask)
            elif cfg.msk_config.sigmoid_mask:
                dep_info.diff_mask, dep_info.out_indices = self._get_oc_rep_sigmoid(input, in_meta, dep_info)
            else:
                dep_info.diff_mask = self._get_oc_rep(input, in_meta, dep_info)
                dep_info.out_indices = self._get_out_indices(input, in_meta, dep_info, dep_info.diff_mask)

        diff_mask = dep_info.diff_mask
        out_order = dep_info.out_indices

        # TODO: Transfer this part to other dynamic layers.
        if not req_soft_fd or cfg.msk_config.iterative_pruning:
            diff_mask_bool = diff_mask.bool()
            diff_mask = diff_mask[diff_mask_bool]
            out_indices = out_order[diff_mask_bool]
        else:
            out_indices = out_order

        # Unified output calculation for both hard/soft forward
        output = self._core_forward(input, in_meta.out_indices, out_indices)

        if ctx._req_hard_forward:
            forward_meta = ctx.ForwardMeta(out_indices)
            if req_soft_fd:
                if use_relu_mask:
                    forward_meta = ctx.ForwardMeta(out_indices, mask=diff_mask[None, :, None, None])
                else:
                    output = output * diff_mask[None, :, None, None]
        else:
            soft_in_channels = in_meta.soft_in_channels
            if soft_in_channels is None:  # Handle the first block conv layer
                soft_in_channels = self.in_channels
            soft_out_channels = dep_info.soft_out_channels
            batch, _, oh, ow = output.size()
            soft_flops, all_flops = FLOPsHelper.infer_mask_flops(
                self, soft_in_channels, soft_out_channels, oh, ow, batch
            )
            ctx._all_soft_flops += soft_flops
            ctx._all_flops += all_flops
            group.gp_flops[self._name] = (soft_flops.item(), all_flops)

            if use_relu_mask:
                forward_meta = ctx.ForwardMeta(
                    out_indices,
                    soft_out_channels,
                    diff_mask[None, :, None, None],
                    self._name,
                )
            else:
                output = output * diff_mask[None, :, None, None]
                forward_meta = ctx.ForwardMeta(out_indices, soft_out_channels)

        if ctx._req_param_stat:
            total_params = self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
            real_params = input.size(1) * output.size(1) * self.kernel_size[0] * self.kernel_size[1] + (
                output.size(1) if self.bias is not None else 0
            )
            # TODO: Better handling group convolution.
            if self.groups != 1:
                real_params /= output.size(1)
            ctx._param_stat[0] += real_params
            ctx._param_stat[1] += total_params

        return TensorWithMeta(output, forward_meta)

    @MultiBindMethod
    def _conv2d_get_oc_rep_sigmoid(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        dep_info: DependencyInfo,
    ) -> tuple[Tensor, Tensor]:
        mask_idx = ctx._cur_subnet.idx
        assert mask_idx is not None, "The mask index should not be None when the output number is not specified."

        gp = cast(ctx.DependencyGroup, self._group)
        mask = gp.get_mask(mask_idx)

        ch_divisor = cfg.msk_config.ceil_divisor
        if ch_divisor == "same":
            ch_divisor = gp.ch_divisor

        if cfg.msk_config.max_init and not gp.order_initialized:
            ctx._mod_mask_init(self, mask, gp, ch_divisor)

        if cfg.msk_config.gp_grad_norm == "t1":
            mask = mask / len(gp)
        elif cfg.msk_config.gp_grad_norm == "t2":
            bp_mask = mask / len(gp)
            mask = torch.sub(mask, bp_mask).detach() + bp_mask

        norm_mask = ctx._gw.watch(F.sigmoid(mask), f"{self._name}<get_oc_rep>: norm_mask")
        # norm_mask_2 = torch.stack([1 - norm_mask_1, norm_mask_1], dim=1)
        # logits = torch.log(norm_mask_2)
        # gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        # gumbels = (logits + gumbels) / cfg.msk_config.softmax_temp
        # norm_mask = gumbels.softmax(1)[:, 1]
        expanded_mask = torch.repeat_interleave(mask, gp.repeat_nums)
        expanded_prob = ctx._gw.watch(
            torch.repeat_interleave(norm_mask, gp.repeat_nums),
            f"{self._name}<get_oc_rep>: expanded_prob",
        )

        # threshold = torch.mean(expanded_prob)
        threshold = torch.mean(expanded_mask)
        # threshold = cast(Tensor, min(0.5, expanded_prob.max()))
        # hard_mask = torch.ge(expanded_prob, threshold)
        hard_mask = torch.ge(expanded_mask, threshold)
        # soft_approx = torch.sum(torch.sigmoid(expanded_prob - threshold))
        soft_approx = torch.sum(torch.sigmoid(expanded_mask - threshold))
        soft_out_channels = (torch.sum(hard_mask) - soft_approx).detach() + soft_approx

        if not ctx._req_hard_forward:
            # Update the global mask L1 norm and entropy
            ctx._all_norm += torch.linalg.norm(mask, ord=1)
            ctx._all_entropy += torch.sum(norm_mask * (1 - norm_mask))

        # Record current proxy
        ctx._cur_subnet.proxy[self._name] = int(soft_out_channels.item())
        ctx._cur_subnet.proxy._cur[ctx._dep_groups.index(gp)] = int(soft_out_channels.item())

        # STE (Straight-through estimator) for the mask
        req_soft_fd = ctx._req_soft_forward or (cfg.msk_config.solidify and not gp.solidified[mask_idx])
        dep_info.soft_out_channels = soft_out_channels
        return (
            ctx._gw.watch(
                (hard_mask.float() - expanded_prob).detach() + expanded_prob,
                f"{self._name}<get_oc_rep>: returned_mask",
            )
            if not req_soft_fd or (cfg.msk_config.straight_solidify and gp.solidified[mask_idx])
            else expanded_prob,
            torch.arange(self.out_channels, device=input.device),
        )

    @MultiBindMethod
    def _conv2d_get_oc_rep(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        dep_info: DependencyInfo,
    ) -> Tensor:
        # mask[0] -> the probability to prune `out_channels - 1` most non-significant channels
        # mask[1] -> the probability to prune `out_channels - 2` most non-significant channels
        # ...
        # mask[out_channels - 2] -> the probability to prune the most non-significant channel
        # mask[out_channels - 1] -> the probability to keep all channels
        mask_idx = ctx._cur_subnet.idx
        assert mask_idx is not None, "The mask index should not be None when the output number is not specified."

        gp = cast(ctx.DependencyGroup, self._group)
        mask = gp.get_mask(mask_idx)

        ch_divisor = cfg.msk_config.ceil_divisor
        if ch_divisor == "same":
            ch_divisor = gp.ch_divisor

        if cfg.msk_config.max_init and not gp.order_initialized:
            ctx._mod_mask_init(self, mask, gp, ch_divisor)

        if cfg.msk_config.gp_grad_norm == "t1":
            mask = mask / len(gp)
        elif cfg.msk_config.gp_grad_norm == "t2":
            bp_mask = mask / len(gp)
            mask = torch.sub(mask, bp_mask).detach() + bp_mask

        # TODO: Temperature from 0.1 to 1.0, initialized using [0, ..., 0, 1]
        norm_mask = ctx._gw.watch(
            F.softmax(torch.div(mask, cfg.msk_config.softmax_temp), dim=0),
            f"{self._name}<get_oc_rep>: norm_mask",
        )
        keep_prob = ctx._gw.watch(
            torch.cumsum(norm_mask.flip(0), 0),
            f"{self._name}<get_oc_rep>: cumsum",
        ).flip(0)
        expanded_prob = ctx._gw.watch(
            torch.repeat_interleave(keep_prob, gp.repeat_nums),
            f"{self._name}<get_oc_rep>: expanded_prob",
        )

        soft_out_channels = ctx.STECeil.apply(
            torch.dot(norm_mask, gp.channel_nums.float()), ch_divisor, self.out_channels
        )
        hard_mask = torch.zeros(self.out_channels, dtype=torch.float32, device=input.device)
        hard_mask[: int(soft_out_channels.item())] = 1.0

        # # Record channel difference
        # soft_out_channels_ori = int(torch.dot(norm_mask, gp.channel_nums.float()).item())
        # hard_out_channels = (expanded_prob >= expanded_prob.mean()).sum().item()
        # cfg.rt_tb_logger.add_scalars(
        #     f"{self._name}/out_channels",
        #     {"max_ch": self.out_channels, "err": abs(soft_out_channels_ori - hard_out_channels)},
        #     cfg.rt_iter,
        # )

        if not ctx._req_hard_forward:
            # Update the global mask L1 norm and entropy
            ctx._all_norm += torch.linalg.norm(mask, ord=1)
            ctx._all_entropy += torch.special.entr(norm_mask).sum()

        # Record current proxy
        ctx._cur_subnet.proxy[self._name] = int(soft_out_channels.item())
        ctx._cur_subnet.proxy._cur[ctx._dep_groups.index(gp)] = int(soft_out_channels.item())

        # STE (Straight-through estimator) for the mask
        req_soft_fd = ctx._req_soft_forward or (cfg.msk_config.solidify and not gp.solidified[mask_idx])
        dep_info.soft_out_channels = soft_out_channels
        return (
            ctx._gw.watch(
                (hard_mask - expanded_prob).detach() + expanded_prob,
                f"{self._name}<get_oc_rep>: returned_mask",
            )
            if not req_soft_fd or (cfg.msk_config.straight_solidify and gp.solidified[mask_idx])
            else expanded_prob
        )

    @MultiBindMethod
    def _conv2d_get_out_indices(
        ctx: __DynamicResNet,
        self: Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        dep_info: DependencyInfo,
        new_out: Tensor,
    ) -> SliceType:
        """
        Return the importance order of the channels. Note that it is not the output indices.
        """
        if cfg.msk_config.trivial_l1_order:
            return torch.arange(self.out_channels, device=input.device)

        gp = cast(ctx.DependencyGroup, self._group)

        fix_l1_order = cfg.msk_config.fix_l1_order
        if fix_l1_order and gp.order_initialized:
            return gp.l1_order

        weight_l1 = torch.stack(
            [_m.weight.detach().abs().sum(dim=tuple(range(1, _m.weight.ndim))) for _m in gp.layers.values()],
            dim=0,
        ).mean(dim=0)
        l1_order = torch.argsort(weight_l1, descending=True)

        if fix_l1_order:
            gp.l1_order.copy_(l1_order)
            gp.order_initialized = torch.ones_like(gp.order_initialized, dtype=torch.bool)

        return l1_order

    def _divide_channel(self: __DynamicResNet, oc: int, ch_divisor: int) -> tuple[Tensor, Tensor]:
        if cfg.msk_config.iterative_pruning:
            avail_iters = (int(cfg.epochs * cfg.msk_config.ip_end_epoch) - cfg.st_warmup_epochs) * len(
                cfg.rt_train_loader
            )
            if (ip_iter_factor := cfg.msk_config.ip_iter_factor) != -1:
                avail_iters = ip_iter_factor * len(cfg.rt_train_loader)
            if len(self._dep_groups) < 1:
                raise ValueError("The number of dependency groups should be greater than 0.")
            assert cfg.msk_config.ip_prune_freq > 0
            avail_prune_nums_per = int(avail_iters / (cfg.msk_config.ip_prune_freq * (len(self._dep_groups) ** 2)))
            prune_factor = 1 / (avail_prune_nums_per + 1)
            if prune_factor == 1:
                warnings.warn(
                    "The pruning factor is 1, which means no pruning is allowed. "
                    "Please consider enlarging the end epoch or reducing the pruning frequency.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if cfg.msk_config.fix_ch_frac:
                prune_factor = cfg.msk_config.ch_frac
            ch_divisor = math.ceil(oc * prune_factor)
            while oc % ch_divisor != 0:
                ch_divisor = min(oc, ch_divisor + 1)
            if cfg.msk_config.ip_dy_div or cfg.msk_config.ip_grad_div:
                ch_divisor = cfg.msk_config.ip_div_range[0]
            if cfg.msk_config.fix_ch_divisor:
                ch_divisor = cfg.msk_config.ch_divisor
        elif cfg.msk_config.auto_divisor:

            def factors(n):
                return set(
                    functools.reduce(
                        list.__add__,
                        ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
                    )
                )

            min_g_num, max_g_num = cfg.msk_config.divisor_range
            avail_factors = list(filter(lambda f: min_g_num <= f <= max_g_num, factors(oc)))
            assert avail_factors, f"No available factors for {oc} in the range of {min_g_num} and {max_g_num}."
            # TODO: Other choices of the divisor.
            chosen_factor = max(avail_factors)
            ch_divisor = oc // chosen_factor
        else:
            # TODO: Better handling the case when `ch_divisor` is not a divisor of `oc`.
            while oc % ch_divisor != 0:
                ch_divisor = max(1, ch_divisor - 1)

        ch_num = list(range(ch_divisor, oc - ch_divisor + 1, ch_divisor)) + [oc]
        return torch.tensor(ch_num), torch.tensor(ch_divisor)

    def _mod_mask_init(
        ctx: __DynamicResNet,
        self: nn.Module,
        mask: Tensor,
        gp: DependencyGroup,
        ch_divisor: int,
    ):
        match cfg.msk_config.max_init_method:
            case "default":
                max_init_temp = cfg.msk_config.softmax_temp
                max_channel = gp.channel_nums[-1]
                if max_channel <= ch_divisor:
                    raise ValueError("The maximum channel number is no more than the ceil divisor.")
                max_init_target = (max_channel - ch_divisor) / max_channel
                mask_len = len(gp.channel_nums)
                init_val = max_init_temp * math.log((mask_len - 1) * max_init_target / (1 - max_init_target))
                new_mask = torch.zeros_like(mask)
                new_mask[-1] = init_val

                if cfg.msk_config.keep_min_norm:
                    match cfg.msk_config.min_norm_type:
                        case 2:
                            new_mask.add_(ctx._mask_optimizer._cal_min_norm(new_mask)[1])
                        case "inf":
                            sorted_mask = new_mask.sort()[0]
                            bias = 0.5 * (sorted_mask[-1] + sorted_mask[0])
                            new_mask = new_mask - bias
                        case _:
                            raise NotImplementedError

                mask.data.copy_(new_mask)

            case "max_entr":
                new_mask = next(
                    iter(
                        ctx._solve_mask_border(
                            gp.channel_nums,
                            gp.channel_nums[-1],
                            ch_divisor,
                            "entr",
                            maximize=True,
                            out_device=mask.device,
                        ).values()
                    )
                )
                mask.data.copy_(new_mask)

            case "min_var":
                new_mask = next(
                    iter(
                        ctx._solve_mask_border(
                            gp.channel_nums,
                            gp.channel_nums[-1],
                            ch_divisor,
                            "var",
                            maximize=False,
                            out_device=mask.device,
                        ).values()
                    )
                )
                mask.data.copy_(new_mask)

            case "min_2_norm":
                new_mask = next(
                    iter(
                        ctx._solve_mask_border(
                            gp.channel_nums,
                            gp.channel_nums[-1],
                            ch_divisor,
                            "norm",
                            2,
                            maximize=False,
                            out_device=mask.device,
                        ).values()
                    )
                )
                mask.data.copy_(new_mask)

            case _:
                raise NotImplementedError

    @staticmethod
    def _decoupled_kl_loss(out: Tensor, soft_out: Tensor, mask_to_backward: list[Tensor]):
        match cfg.msk_config.decoupled_kl_type:
            case "t1":
                sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out, dim=1), F.softmax(soft_out.detach(), dim=1)
                )
                sup_loss.backward()
                sup_loss_2 = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out.detach(), dim=1), F.softmax(soft_out, dim=1)
                )
                sup_loss_2.backward(inputs=mask_to_backward) if mask_to_backward else None
            case "t2":
                sup_loss = (
                    0.5
                    * cfg.msk_config.aux_loss_coef
                    * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(out, dim=1), F.softmax(soft_out, dim=1))
                )
                sup_loss.backward()
                sup_loss_2 = sup_loss
            case "t3":
                sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out, dim=1), F.softmax(soft_out.detach(), dim=1)
                )
                sup_loss.backward()
                sup_loss_2 = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(soft_out, dim=1), F.softmax(out.detach(), dim=1)
                )
                sup_loss_2.backward(inputs=mask_to_backward) if mask_to_backward else None
            case "t4":
                sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out, dim=1), F.softmax(soft_out.detach(), dim=1)
                )
                sup_loss_2 = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(soft_out, dim=1), F.softmax(out.detach(), dim=1)
                )
                (0.5 * (sup_loss + sup_loss_2)).backward()
            case "t5":
                sup_loss = (
                    0.5
                    * cfg.msk_config.aux_loss_coef
                    * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(soft_out, dim=1), F.softmax(out, dim=1))
                )
                sup_loss.backward()
                sup_loss_2 = sup_loss
            case "t6":
                sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(soft_out.detach(), dim=1), F.softmax(out, dim=1)
                )
                sup_loss.backward()
                sup_loss_2 = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(soft_out, dim=1), F.softmax(out.detach(), dim=1)
                )
                sup_loss_2.backward(inputs=mask_to_backward) if mask_to_backward else None
            case _:
                raise NotImplementedError

        sup_loss = 0.5 * (sup_loss + sup_loss_2)
        return sup_loss

    @staticmethod
    def _solve_mask_border(
        ch_nums: Iterable[int] | Tensor | np.ndarray,
        target: Iterable[int] | int | Tensor | np.ndarray,
        ch_divisor: int,
        optim_type: Literal["norm", "var", "entr"] = "norm",
        norm_type: int | Literal["nuc", "fro", "inf"] = 2,
        maximize: bool = False,
        eps: float = 1e-12,
        out_device: Optional[torch.device] = None,
    ) -> dict[int, Tensor]:
        # Convert `ch_nums` to numpy array if necessary.
        if isinstance(ch_nums, Tensor):
            ch_nums = ch_nums.cpu().numpy()
        elif not isinstance(ch_nums, np.ndarray):
            try:
                ch_nums = np.array(ch_nums)
            except Exception as e:
                raise TypeError(
                    "Cannot convert the `ch_nums` to a numpy array, please use an `Iterable` of `int` instead."
                ) from e
        assert isinstance(ch_nums, np.ndarray), (
            "The channel numbers should be an iterable of integers, a numpy array or a Pytorch `Tensor`."
        )

        # Verify the `target` w.r.t. `ch_nums` and `ch_divisor` and resolve the upper/lower bound of the target.
        def _convert_target(target_: int) -> list[float, float]:
            assert ch_nums[0] <= target_ <= ch_nums[-1], "The target is out of the range of the channel numbers."
            assert (target_ - ch_nums[0]) % ch_divisor == 0, (
                "The target is not a multiple of the ceil divisor w.r.t. the minimum channel number."
            )
            z1 = next(itertools.dropwhile(lambda _x1x2: _x1x2[1] < target_, itertools.pairwise(ch_nums)))[0]
            if z1 != ch_nums[0]:
                z1 += eps
            return [z1, target_]

        if isinstance(target, Tensor):
            target = target.to(dtype=torch.int64).tolist()
        elif isinstance(target, np.ndarray):
            target = target.astype(np.int64).tolist()
        if isinstance(target, int):
            target = [target]
        assert isinstance(target, list), (
            "The target should be an integer, an iterable of integers, a numpy array, or a Pytorch `Tensor`."
        )
        zs = map(_convert_target, target)

        # Build the min-norm problem.
        x = cp.Variable(len(ch_nums))
        z = cp.Parameter(2, nonneg=True)

        match optim_type:
            case "norm":
                obj = cp.norm(x, norm_type)
            case "var":
                obj = cp.var(x)
            case "entr":
                obj = cp.sum(cp.entr(x))
            case _:
                raise NotImplementedError

        obj = cp.Minimize(obj) if not maximize else cp.Maximize(obj)
        constraints = [
            cp.sum(x) == 1,
            cp.sum(cp.multiply(x, ch_nums)) >= z[0],
            cp.sum(cp.multiply(x, ch_nums)) <= z[1],
            x >= 0,
            x <= 1,
        ]
        prob = cp.Problem(obj, constraints)

        # Solve the problem for each `z` and store the solution in a dictionary.
        def _solver(z_: list[float, float]) -> Tensor:
            z.value = z_
            prob.solve()
            if prob.status != cp.OPTIMAL:
                warnings.warn(
                    f"The min-norm problem:\n{prob!s}\n"
                    f"is not solved optimally for the target `{z_}` with final status `{prob.status}`.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            softmax_mask = x.value
            assert isinstance(softmax_mask, np.ndarray)
            mask = torch.log(torch.from_numpy(softmax_mask).to(device=out_device, dtype=torch.float32) + eps)
            return mask

        if out_device is None:
            out_device = torch.device(torch.cuda.current_device())

        return {_t: _solver(_z) for _t, _z in zip(target, zs)}

    @exec_range(fn_range=lambda _ep: _ep == cfg.st_warmup_epochs, ep_only_once=True)
    @isolate_random(cfg)
    def __pre_batch_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        cur_generated = len(ctx._pool)
        if cur_generated >= cfg.msk_config.pool_size:
            return

        progressbar = tqdm(range(cfg.msk_config.pool_size), desc="Generating anchors", leave=False)

        if (
            not cfg.msk_config.ip_unified
            and cfg.msk_config.multi_target
            and cfg.msk_config.pool_size != len(cfg.msk_config.multi_target)
        ):
            raise ValueError(
                "The number of FLOPs targets should be the same as the pool size, "
                f"but got {len(cfg.msk_config.multi_target)} and {cfg.msk_config.pool_size} respectively."
            )

        for i in range(cfg.msk_config.pool_size):
            for gp in ctx._dep_groups:
                gp = cast(ctx.DependencyGroup, gp)
                # noinspection PyTypeChecker
                gp.add_mask(i, gp.init_mask(cfg.msk_config.init_type), ctx._mask_optimizer)

            ctx._pool.append(ctx.SubnetInfo(i, ctx.Proxy(), cfg.msk_config.history_num))

        progressbar.close()

        if cfg.msk_config.save_init_pool and cfg.local_rank == 0:
            torch.save(
                ctx.pre_save_checkpoint({}),
                os.path.join(cfg.log_dir, "init_pool.pth.tar"),
            )

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw:
            # Set `all_flops` as a fixed value.
            is_transformer = (
                "vit" in cfg.model
                or "swin" in cfg.model
                or "pit" in cfg.model
                or "t2t" in cfg.model
                or "cait" in cfg.model
                or "bnit" in cfg.model
            )
            img_size = cfg.img_size if is_transformer else 224
            with torch.no_grad():
                device = cast(ctx.DependencyGroup, ctx._dep_groups[0]).channel_nums.device
                rand_in = torch.randn(1, 3, img_size, img_size, device=device)
                cur_subnet = np.array(
                    [len(cast(ctx.DependencyGroup, gp).channel_nums) - 1 for gp in ctx._dep_groups],
                    dtype=np.int64,
                )
                ctx._cur_depth = tuple(ctx.stage_list)
                ctx.set_running_subnet(proxy=cur_subnet)
                cfg.rt_model(rand_in)
                ctx._subnet_sampler.all_flops[(img_size, img_size)] = ctx._all_flops
                ctx._reset_accumulator()
                ctx.reset_running_subnet()

            if not ctx._subnet_sampler.dw_pool:
                if cfg.msk_config.ip_dw_st:
                    all_stage_cb = list(
                        itertools.product(*[range(max(1, int(sl) - 1), int(sl) + 1) for sl in ctx.stage_list])
                    )
                else:
                    all_stage_cb = list(itertools.product(*[range(1, int(sl) + 1) for sl in ctx.stage_list]))

                if not cfg.msk_config.ip_dw_all:
                    if cfg.msk_config.ip_dw_fixed:
                        if cfg.model == "swin":
                            selected_stages = [
                                (2, 3, 2),
                                (2, 4, 2),
                                (2, 5, 2),
                                (2, 4, 3),
                                (1, 3, 2),
                                (1, 3, 3),
                                (1, 5, 2),
                                (1, 4, 2),
                            ]
                        elif cfg.model == "vit_9":
                            selected_stages = [
                                (2, 2, 2),
                                (2, 2, 3),
                                (3, 2, 2),
                            ]
                        else:
                            raise NotImplementedError
                    else:
                        if os.environ.get("ISTP_AB_INIT_SPACE_MAX_DP"):
                            selected_stages = [len(all_stage_cb) - 1]
                        else:
                            selected_stages = broadcast_one_object(
                                np.random.choice(
                                    range(len(all_stage_cb)),
                                    size=cfg.msk_config.ip_pool_size,
                                    replace=False,
                                ),
                                0,
                                cfg.is_ddp,
                            ).tolist()
                        selected_stages = [all_stage_cb[_] for _ in selected_stages]
                    for ss in selected_stages:
                        # TODO: Check whether maximum FLOPs exceeds the target.
                        ctx._subnet_sampler.dw_pool[ss] = ctx.CircleSampler.PoolAttr(
                            ctx._base_subnet.copy(), 0, 0, None, 1.0, False
                        )
                else:
                    selected_stages = all_stage_cb
                    for ss in selected_stages:
                        ctx._cur_depth = ss
                        ctx.set_running_subnet(proxy=cur_subnet)
                        cfg.rt_model(rand_in)
                        all_flops = ctx._all_flops
                        ctx._reset_accumulator()
                        ctx.reset_running_subnet()
                        if (
                            all_flops / ctx._subnet_sampler.all_flops[(img_size, img_size)]
                        ) < cfg.msk_config.target_flops:
                            continue
                        ctx._subnet_sampler.dw_pool[ss] = ctx.CircleSampler.PoolAttr(
                            ctx._base_subnet.copy(), 0, 0, None, 1.0, False
                        )

            # FLOPs-based or Params-based recalibration for `channel_nums` and `ch_divisor`.
            if cfg.msk_config.ip_dw_indie_ch and cfg.msk_config.recalc_divisor != "null":
                global GB_MAKE_DIVISIBLE_TENSOR_DIVISOR
                GB_MAKE_DIVISIBLE_TENSOR_DIVISOR = 1

                for dp, dp_attr in ctx._subnet_sampler.dw_pool.items():
                    if dp_attr.channel_nums:
                        if len(dp_attr.channel_nums) != len(dp_attr.ch_divisor):
                            raise AssertionError
                        continue
                    diff_list_flops = []
                    diff_list_params = []
                    base_subnet = dp_attr.arch
                    for cur_idx in range(len(base_subnet)):
                        cur_subnet = base_subnet.copy()
                        cur_subnet[cur_idx] -= 1
                        ctx._cur_depth = dp
                        ctx.set_running_subnet(cur_subnet)
                        with ctx._get_param_stat():
                            cfg.rt_model(rand_in)
                        diff_list_flops.append(max(0, ctx._all_flops - ctx._all_soft_flops.item()))
                        dp_attr.flops_reduce.append(
                            diff_list_flops[-1] / ctx._subnet_sampler.all_flops[(img_size, img_size)]
                        )
                        diff_list_params.append(ctx._param_stat[1] - ctx._param_stat[0])
                        ctx._reset_accumulator()

                    match cfg.msk_config.recalc_divisor:
                        case "flops":
                            diff_list = diff_list_flops
                        case "params":
                            diff_list = diff_list_params
                        case _:
                            raise NotImplementedError

                    max_idx = np.argmax(diff_list)
                    max_diff = diff_list[max_idx]
                    width_mask = np.array([True] * len(diff_list))

                    def _handle_zero(_m_d: float, _d: float, _i: int):
                        try:
                            return _m_d / _d
                        except ZeroDivisionError:
                            width_mask[_i] = False
                            return cast(ctx.DependencyGroup, ctx._dep_groups[_i]).channel_nums[-1].item()

                    normalized_diff = [_handle_zero(max_diff, diff, i) for i, diff in enumerate(diff_list)]
                    dp_attr.wd_mask = width_mask

                    # TODO: get the scaled_diff
                    ori_divisor = [cast(ctx.DependencyGroup, gp).ch_divisor for gp in ctx._dep_groups]
                    base_divisor = ori_divisor[max_idx].item()
                    ch_scale = cfg.msk_config.ip_recalc_scale
                    scaled_diff = [
                        (base_divisor * ch_scale) if i == max_idx else (base_divisor * normalized_diff[i] * ch_scale)
                        for i in range(len(ori_divisor))
                    ]

                    def _get_new_ch_nums_v1(md_: int, _div_: int | float) -> Tensor:
                        if cfg.msk_config.extreme_merge:
                            return torch.arange(cfg.msk_config.min_width, 1.0 + _div_, _div_)
                        else:
                            min_ch = int(md_ * min(1.0, cfg.msk_config.min_width))
                            _div_ = int(_div_)
                            if (min_ch + _div_) <= md_:
                                rt = torch.arange(min_ch, md_ + 1, _div_, device=device)
                                if rt[-1] < md_:
                                    rt = torch.cat([rt, torch.tensor([md_], device=device)])
                                return rt
                            else:
                                return torch.tensor([min_ch, md_], device=device)

                    def _get_new_ch_nums(md_: int, _div_: int | float) -> tuple[Tensor, int | float]:
                        if cfg.msk_config.extreme_merge:
                            # TODO: Revise to the downward correct case.
                            return torch.arange(cfg.msk_config.min_width, 1.0 + _div_, _div_), _div_
                        else:
                            base_ = md_
                            _div_ = max(min(round(_div_), base_ - 1), 1)
                            min2_ = min_ = max(min(int(md_ * cfg.msk_config.min_width), base_), 1)
                            while ((base_ - min2_) % _div_) != 0:
                                if min2_ == 1:
                                    min2_ = base_
                                    break
                                min2_ -= 1
                            while min2_ < min_:
                                min2_ += _div_
                            min_ = max(min(min2_, base_), 1)
                            return torch.cat(
                                [
                                    torch.arange(min_, base_ + 1, _div_, device=device),
                                    torch.arange(base_ + _div_, md_, _div_, device=device)
                                    if ((base_ + _div_) <= (md_ - 1))
                                    else torch.tensor([], device=device, dtype=torch.int),
                                    torch.arange(md_, md_ + 1, device=device)
                                    if base_ < md_
                                    else torch.tensor([], device=device, dtype=torch.int),
                                ]
                            ), int(_div_)

                    for i, gp in enumerate(ctx._dep_groups):
                        gp = cast(ctx.DependencyGroup, gp)
                        new_ch_nums, new_div = _get_new_ch_nums(gp.channel_nums[-1].item(), scaled_diff[i])
                        dp_attr.channel_nums.append(new_ch_nums)
                        dp_attr.ch_divisor.append(new_div)
                    dp_attr.flops_reduce = [
                        (_fr * _post_d / _pre_d).item()
                        for _fr, _pre_d, _post_d in zip(dp_attr.flops_reduce, ori_divisor, dp_attr.ch_divisor)
                    ]

                    dp_attr.arch = np.array(
                        [
                            np.ceil((len(dp_attr.channel_nums[i]) - 1) * cfg.msk_config.ip_start_ratio)
                            for i, gp in enumerate(ctx._dep_groups)
                        ],
                        dtype=np.int64,
                    )

                    divisor_to_print = " ".join(
                        f"{int(ch_divisor)}/{int(channel_nums[-1])}"
                        for ch_divisor, channel_nums in zip(dp_attr.ch_divisor, dp_attr.channel_nums)
                    )
                    cfg.rt_logger.info(f"The adjusted divisor/channel for depth {dp}: <{divisor_to_print}>")

                    ctx.reset_running_subnet()

                GB_MAKE_DIVISIBLE_TENSOR_DIVISOR = None

            return

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_search_div:
            assert cfg.msk_config.recalc_divisor == "null"
            assert cfg.st_auto_t
            assert not cfg.is_ddp

            images = l_dict["images"].cuda(non_blocking=True)
            l_metric = torch.nn.KLDivLoss(reduction="batchmean")
            optimizer = l_dict["optimizer"]
            ctx.reset_running_subnet()

            manual_div = [4, 4, 4, 4, 2, 16, 4, 4, 4]
            device = cast(ctx.DependencyGroup, ctx._dep_groups[0]).channel_nums.device

            def _get_new_ch_nums(md_: int, div_: int, base_: int) -> tuple[Tensor, int]:
                re_ = base_ % div_
                min_ = min(base_, re_ + div_)
                new_ch_ = torch.cat(
                    [
                        torch.arange(min_, base_ + 1, div_, device=device),
                        torch.arange(base_ + div_, md_ - div_ + 1, div_, device=device)
                        if ((base_ + div_) <= (md_ - div_))
                        else torch.tensor([], device=device),
                        torch.arange(md_, md_ + 1, device=device),
                    ]
                )
                new_idx_ = new_ch_.tolist().index(base_)
                return new_ch_, new_idx_

            for i, (div_, gp) in enumerate(zip(manual_div, ctx._dep_groups)):
                gp = cast(ctx.DependencyGroup, gp)
                gp.channel_nums, ctx._base_subnet[i] = _get_new_ch_nums(
                    gp.channel_nums[-1].item(),
                    div_,
                    gp.channel_nums[ctx._base_subnet[i]].item(),
                )
                gp.ch_divisor.copy_(div_)

            with torch.no_grad():
                max_out = cfg.rt_model(images)
                normalized_shape = (max_out.size()[1],)
                max_out_auto_t = F.softmax(F.layer_norm(max_out, normalized_shape, None, None, 1e-7), dim=1)

            sup_loss_list = []
            for subnet_idx in range(len(ctx._dep_groups)):
                optimizer.zero_grad()
                ctx.set_running_subnet(subnet_idx)
                out = cfg.rt_model(images)
                out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                sup_loss = l_metric(F.log_softmax(out_auto_t, dim=1), max_out_auto_t)
                sup_loss.backward()
                sup_loss = torch.abs(torch.cat([p.grad.flatten() for p in cfg.rt_model.parameters() if p is not None]))
                sup_loss = sup_loss.sum() / torch.nonzero(sup_loss).size(0)
                sup_loss_list.append(sup_loss.item())

            # TODO: Finish the rest of the code.
            raise

        if cfg.msk_config.iterative_pruning and cfg.msk_config.recalc_divisor != "null":
            # TODO: Resume may be buggy due to the states of `ch_divisor` and `channel_nums`.
            with torch.no_grad():
                device = cast(ctx.DependencyGroup, ctx._dep_groups[0]).channel_nums.device
                base_subnet = torch.tensor(
                    [cast(ctx.DependencyGroup, _gp).channel_nums[-1] for _gp in ctx._dep_groups],
                    device=device,
                )

                is_transformer = (
                    "vit" in cfg.model
                    or "swin" in cfg.model
                    or "pit" in cfg.model
                    or "t2t" in cfg.model
                    or "cait" in cfg.model
                    or "bnit" in cfg.model
                )
                if is_transformer:
                    rand_in = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
                else:
                    rand_in = torch.randn(1, 3, 224, 224, device=device)

                diff_list_flops = []
                diff_list_params = []
                for cur_idx in range(len(base_subnet)):
                    cur_subnet = base_subnet.clone()
                    cur_subnet[cur_idx] -= 1
                    ctx.set_running_subnet(cur_subnet)
                    with ctx._get_param_stat():
                        cfg.rt_model(rand_in)
                    diff_list_flops.append((ctx._all_flops - ctx._all_soft_flops).item())
                    diff_list_params.append(ctx._param_stat[1] - ctx._param_stat[0])
                    ctx._reset_accumulator()

                match cfg.msk_config.recalc_divisor:
                    case "flops":
                        diff_list = diff_list_flops
                    case "params":
                        diff_list = diff_list_params
                    case _:
                        raise NotImplementedError

                ori_units = sum(len(cast(ctx.DependencyGroup, gp).channel_nums) for gp in ctx._dep_groups)
                max_diff = max(diff_list)
                normalized_diff = [max_diff / diff for diff in diff_list]
                # TODO: Better handling the scale problem.
                units = sum(
                    cast(ctx.DependencyGroup, gp).channel_nums[-1].item() / diff
                    for gp, diff in zip(ctx._dep_groups, normalized_diff)
                )
                diff_scale = np.ceil(units / ori_units)
                # TODO: To remove
                diff_scale = 1.0
                scaled_diff = [round(diff * diff_scale) for diff in normalized_diff]

                def _get_new_ch_nums(md_: int, div_: int) -> Tensor:
                    re_ = md_ % div_
                    min_ = re_ + div_
                    return torch.arange(min_, md_ + 1, div_, device=device)

                for i, gp in enumerate(ctx._dep_groups):
                    gp = cast(ctx.DependencyGroup, gp)
                    gp.channel_nums = _get_new_ch_nums(gp.channel_nums[-1].item(), scaled_diff[i])
                    gp.ch_divisor.copy_(scaled_diff[i])

                divisor_to_print = " ".join(
                    f"{int(gp.ch_divisor)}/{int(gp.channel_nums[-1])}"
                    for gp in cast(list[ctx.DependencyGroup], ctx._dep_groups)
                )
                cfg.rt_logger.info(f"The adjusted divisor/channel of groups: <{divisor_to_print}>")
                ctx._base_subnet = np.array(
                    [
                        np.ceil((len(cast(ctx.DependencyGroup, gp).channel_nums) - 1) * cfg.msk_config.ip_start_ratio)
                        for gp in ctx._dep_groups
                    ],
                    dtype=np.int64,
                )
                ctx.reset_running_subnet()

    @exec_range(fn_range=lambda _ep: _ep < cfg.msk_config.hard_ft_epoch)
    def __student_pre_backward_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        assert ctx._all_flops > 0, "Nothing gathered."

        sup_losses: AverageMeter = l_dict["sup_losses"]
        b_size = l_dict["out"].size(0)

        if cfg.msk_config.ip_dw:
            img_size = cast(tuple[int, int], tuple(l_dict["input_student"].size()[-2:]))
            if img_size not in ctx._subnet_sampler.all_flops:
                tmp_all_soft_flops = ctx._all_soft_flops
                tmp_all_flops = ctx._all_flops
                tmp_all_norm = ctx._all_norm
                tmp_all_entropy = ctx._all_entropy
                tmp_param_stat = ctx._param_stat
                tmp_dp = ctx._cur_depth
                tmp_cur_subnet = ctx._cur_subnet
                ctx._reset_accumulator()

                ctx._cur_depth = tuple(ctx.stage_list)
                device = cast(ctx.DependencyGroup, ctx._dep_groups[0]).channel_nums.device
                rand_in = torch.randn(1, 3, *img_size, device=device)
                cur_subnet = np.array(
                    [len(ctx._subnet_sampler._get_ch_nums(gp, i)) - 1 for i, gp in enumerate(ctx._dep_groups)],
                    dtype=np.int64,
                )
                ctx.set_running_subnet(proxy=cur_subnet)
                cfg.rt_model(rand_in)
                ctx._subnet_sampler.all_flops[img_size] = ctx._all_flops
                ctx._reset_accumulator()
                ctx.reset_running_subnet()

                ctx._cur_depth = tmp_dp
                ctx._all_soft_flops = tmp_all_soft_flops
                ctx._all_flops = tmp_all_flops
                ctx._all_norm = tmp_all_norm
                ctx._all_entropy = tmp_all_entropy
                ctx._param_stat = tmp_param_stat
                ctx._cur_subnet = tmp_cur_subnet

            ctx._all_flops = ctx._subnet_sampler.all_flops[img_size]
            ctx._all_soft_flops = ctx._all_soft_flops / b_size

        # Calculate FLOPs loss
        if cfg.msk_config.flops_loss_coef > 0:
            assert not cfg.msk_config.iterative_pruning
            target_flops = (
                cfg.msk_config.multi_target[ctx._cur_subnet.idx]
                if cfg.msk_config.multi_target
                else cfg.msk_config.target_flops
            )
            flops_loss = cast(
                Tensor,
                cfg.msk_config.flops_loss_coef * (ctx._all_soft_flops / ctx._all_flops - target_flops) ** 2,
            )
            flops_loss.backward(retain_graph=True) if flops_loss.requires_grad else None
        else:
            flops_loss = cast(Tensor, ctx._all_soft_flops / ctx._all_flops).detach()

        if not cfg.msk_config.iterative_pruning:
            ctx._mask_optimizer.prep_step(flops_loss, sup_losses.val)
            ctx._mask_optimizer.store_grad(ctx._cur_subnet.idx, "flops_grad")

        if cfg.msk_config.iterative_pruning and cfg.rt_epoch >= (cfg.st_warmup_epochs + cfg.msk_config.ip_warm_up):
            if not (
                (
                    ctx._flops_satisfied
                    if not cfg.msk_config.ip_dw
                    else ctx._subnet_sampler.dw_pool[ctx._cur_depth].flops_satisfied
                )
                and (not cfg.msk_config.ip_reverse or cfg.rt_epoch >= (cfg.epochs * cfg.msk_config.ip_end_epoch))
            ):
                chosen_gp = cast(ctx.DependencyGroup, ctx._dep_groups[ctx._cur_subnet.idx])

                if cfg.msk_config.ip_dw:
                    cur_pool = ctx._cur_depth
                    chosen_gp.sub_flops[cur_pool] = flops_loss.item()
                    # ctx._subnet_sampler.dw_pool[cur_pool].base_flops_ratio = flops_loss.item()
                    ctx._subnet_sampler.dw_pool[cur_pool].dis_metric.update(sup_losses.val, b_size)
                    ctx._subnet_sampler.dw_pool[cur_pool].dis_metric.sync_metrics()
                    if "dis" in cfg.msk_config.ip_metric:
                        chosen_gp.prune_metric["dis"][cur_pool].update(sup_losses.val, b_size)
                        chosen_gp.prune_metric["dis"][cur_pool].sync_metrics()
                        if cfg.msk_config.ip_broadcast:
                            ctx._subnet_sampler.broadcast_metric(
                                chosen_gp.prune_metric["dis"],
                                cur_pool,
                                sup_losses.val,
                                b_size,
                            )
                    if "gt" in cfg.msk_config.ip_metric:
                        tar_losses: AverageMeter = l_dict["tar_losses"]
                        chosen_gp.prune_metric["gt"][cur_pool].update(tar_losses.val, b_size)
                        chosen_gp.prune_metric["gt"][cur_pool].sync_metrics()
                        if cfg.msk_config.ip_broadcast:
                            ctx._subnet_sampler.broadcast_metric(
                                chosen_gp.prune_metric["gt"],
                                cur_pool,
                                tar_losses.val,
                                b_size,
                            )
                    if "acc" in cfg.msk_config.ip_metric:
                        out: Tensor = l_dict["out"]
                        target: Tensor = l_dict["target"]
                        acc = accuracy(output=out.detach().data, target=target, topk=(1,))[0]
                        chosen_gp.prune_metric["acc"][cur_pool].update(acc.item(), b_size)
                        chosen_gp.prune_metric["acc"][cur_pool].sync_metrics()
                        if cfg.msk_config.ip_broadcast:
                            ctx._subnet_sampler.broadcast_metric(
                                chosen_gp.prune_metric["acc"],
                                cur_pool,
                                acc.item(),
                                b_size,
                            )
                else:
                    chosen_gp.sub_flops = flops_loss.item()
                    cp = cast(dict[str, AverageMeter], chosen_gp.prune_metric)
                    if "dis" in cfg.msk_config.ip_metric:
                        cp["dis"].update(sup_losses.val, b_size)
                        cp["dis"].sync_metrics()
                    if "gt" in cfg.msk_config.ip_metric:
                        tar_losses: AverageMeter = l_dict["tar_losses"]
                        cp["gt"].update(tar_losses.val, b_size)
                        cp["gt"].sync_metrics()
                    if "acc" in cfg.msk_config.ip_metric:
                        out: Tensor = l_dict["out"]
                        target: Tensor = l_dict["target"]
                        acc = accuracy(output=out.detach().data, target=target, topk=(1,))[0]
                        cp["acc"].update(acc.item(), b_size)
                        cp["acc"].sync_metrics()

        if cfg.msk_config.iterative_pruning:
            ctx._all_norm = torch.tensor(0.0)
            ctx._all_entropy = torch.tensor(0.0)

        # Calculate L1 norm loss
        if cfg.msk_config.norm_loss_coef > 0:
            assert not cfg.msk_config.iterative_pruning
            norm_loss = cast(Tensor, cfg.msk_config.norm_loss_coef * ctx._all_norm)
            norm_loss.backward(retain_graph=True) if norm_loss.requires_grad else None
            ctx._mask_optimizer.store_grad(ctx._cur_subnet.idx, "norm_grad")
        else:
            norm_loss = ctx._all_norm

        # Calculate entropy loss
        if cfg.msk_config.entropy_loss_coef > 0:
            assert not cfg.msk_config.iterative_pruning
            entropy_loss = cast(Tensor, cfg.msk_config.entropy_loss_coef * ctx._all_entropy)
            entropy_loss.backward(retain_graph=True) if entropy_loss.requires_grad else None
            ctx._mask_optimizer.store_grad(ctx._cur_subnet.idx, "entropy_grad")
        else:
            entropy_loss = ctx._all_entropy

        # Metrics accumulation
        ctx._flops_losses.update(flops_loss.item(), b_size)
        ctx._norm_losses.update(norm_loss.item(), b_size)
        ctx._entropy_losses.update(entropy_loss.item(), b_size)

        # Log string setup
        cfg.rt_tmp_var.extra_log += f"Subnet_Idx {ctx._cur_subnet.idx}\t"
        cfg.rt_tmp_var.extra_log += f"FLOPs_Loss {ctx._flops_losses.val:.4f} ({ctx._flops_losses.avg:.4f})\t"
        cfg.rt_tmp_var.extra_log += f"Norm_Loss {norm_loss:.4f} ({ctx._norm_losses.avg:.4f})\t"
        cfg.rt_tmp_var.extra_log += f"Entropy_Loss {entropy_loss:.4f} ({ctx._entropy_losses.avg:.4f})\t"

        # Optional logging for optimization time
        optim_timer = ctx._timer["optim_time"]
        cfg.rt_tmp_var.extra_log += f"Optim_Time {optim_timer.val:.4f} ({optim_timer.avg:.4f})\t"

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw:
            cfg.rt_tmp_var.extra_log += f"Depth {ctx._cur_depth}\t"

            if cfg.msk_config.ip_dw_all:
                cfg.rt_tmp_var.extra_log += f"Pool_Size {len(ctx._subnet_sampler.dw_pool)}\t"

        # Clean up gathered FLOPs and norm
        ctx._reset_accumulator()

    def __post_student_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        if cfg.msk_config.iterative_pruning and not cfg.msk_config.ip_unified and not cfg.msk_config.ip_sup_net:
            return

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_sup_net:
            max_output = l_dict["max_output"]
            images = l_dict["images"]
            is_transformer = l_dict["is_transformer"]
            model = l_dict["model"]
            cur_subnet = ctx._base_subnet
            dep_groups: list[ctx.DependencyGroup] = ctx._dep_groups

            resolution = (
                broadcast_one_object(cfg.resos[random.randint(0, len(cfg.resos) - 1)], 0, cfg.is_ddp)
                if not is_transformer
                else cfg.img_size
            )
            input_student = F.interpolate(images, (resolution, resolution), mode="bilinear", align_corners=True)

            if cfg.msk_config.ip_dw:
                cur_subnet = ctx._subnet_sampler.dw_pool[ctx._cur_depth].arch

            if not cfg.msk_config.ip_new_sup:
                cur_subnet = np.array(
                    [
                        min(
                            make_divisible(
                                np.random.randint(
                                    base, len(ch_nums := ctx._subnet_sampler._get_ch_nums(gp, i, ctx._cur_depth))
                                ),
                                int(cfg.msk_config.ip_sup_min_div),
                            ),
                            (len(ch_nums) - 1),
                        )
                        for i, (base, gp) in enumerate(zip(cur_subnet, dep_groups))
                    ]
                )
            else:
                cur_subnet = torch.tensor(
                    [
                        min(
                            make_divisible(
                                np.random.randint(
                                    (ch_nums := ctx._subnet_sampler._get_ch_nums(gp, i, ctx._cur_depth))[base].item(),
                                    (max_ch_num := ch_nums[-1].item()) + 1,
                                ),
                                int(max_ch_num * cfg.msk_config.ip_sup_min_div),
                            ),
                            max_ch_num,
                        )
                        for i, (base, gp) in enumerate(zip(cur_subnet, dep_groups))
                    ],
                    dtype=torch.int64,
                )

            if cfg.msk_config.ip_dw:
                cur_depth = tuple(
                    np.random.randint(depth, ctx.stage_list[i] + 1) for i, depth in enumerate(ctx._cur_depth)
                )
                ctx._cur_depth = cur_depth

            model.module.set_running_subnet(cur_subnet)
            out = model(input_student)

            if not cfg.st_auto_t:
                # noinspection PyUnboundLocalVariable
                sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out, dim=1), F.softmax(max_output.detach(), dim=1)
                )
            else:
                normalized_shape = (out.size()[1],)
                max_output_auto_t = F.layer_norm(max_output.detach(), normalized_shape, None, None, 1e-7)
                out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(out_auto_t, dim=1),
                    F.softmax(max_output_auto_t, dim=1),
                )

            ctx._aux_losses.update(sup_loss.item(), images.size(0))
            cfg.rt_tmp_var.extra_log += f"Aux_Loss {ctx._aux_losses.val:.4f} ({ctx._aux_losses.avg:.4f})\t"

            sup_loss.backward()
            ctx._reset_accumulator()

            return

        if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_unified:
            max_output = l_dict["max_output"]
            images = l_dict["images"]
            is_transformer = l_dict["is_transformer"]
            model = l_dict["model"]

            for cur_subnet in ctx._subnet_sampler.target_subnets.values():
                resolution = (
                    broadcast_one_object(cfg.resos[random.randint(0, len(cfg.resos) - 1)], 0, cfg.is_ddp)
                    if not is_transformer
                    else cfg.img_size
                )
                input_student = F.interpolate(
                    images,
                    (resolution, resolution),
                    mode="bilinear",
                    align_corners=True,
                )

                model.module.set_running_subnet(cur_subnet)
                out = model(input_student)

                if not cfg.st_auto_t:
                    # noinspection PyUnboundLocalVariable
                    sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(out, dim=1), F.softmax(max_output.detach(), dim=1)
                    )
                else:
                    normalized_shape = (out.size()[1],)
                    max_output_auto_t = F.layer_norm(max_output.detach(), normalized_shape, None, None, 1e-7)
                    out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                    sup_loss = cfg.st_sup_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(out_auto_t, dim=1),
                        F.softmax(max_output_auto_t, dim=1),
                    )

                # TODO: Logger for the additional `sup_loss`.
                sup_loss.backward()
                ctx._reset_accumulator()

            return

        soft_out = l_dict["out"]
        images = l_dict["images"]
        chosen_subnet: ctx.SubnetInfo = l_dict["chosen_subnet"]
        chosen_subnet = broadcast_one_object(chosen_subnet, 0, cfg.is_ddp)

        mask_idx = chosen_subnet.idx
        solidify = cfg.msk_config.solidify
        straight_solidify = cfg.msk_config.straight_solidify
        solidify_grad = cfg.msk_config.solidify_grad

        # Forward again, however, in hard mode and disable soft forward.
        ctx._req_hard_forward = True
        ctx._req_soft_forward = False
        masks_list = chosen_subnet.masks if (not solidify or solidify_grad) else chosen_subnet.non_solidified_masks
        if not cfg.msk_config.decoupled_kl:
            for msk in masks_list:
                msk.requires_grad = False

        all_solidified = False
        none_solidified = False
        if solidify:
            groups: list[ctx.DependencyGroup] = ctx._dep_groups
            all_solidified = all([gp.solidified[mask_idx] for gp in groups])
            none_solidified = all([not gp.solidified[mask_idx] for gp in groups])

        if not none_solidified and not straight_solidify:
            is_transformer = (
                "vit" in cfg.model
                or "swin" in cfg.model
                or "pit" in cfg.model
                or "t2t" in cfg.model
                or "cait" in cfg.model
                or "bnit" in cfg.model
            )
            resolution = (
                broadcast_one_object(cfg.resos[random.randint(0, len(cfg.resos) - 1)], 0, cfg.is_ddp)
                if not is_transformer
                else cfg.img_size
            )
            input_student = F.interpolate(images, (resolution, resolution), mode="bilinear", align_corners=True)

            ctx.set_running_subnet(chosen_subnet)
            # TakeNote: It is quite important to use `cfg.rt_model` rather than simply `ctx`, as the latter one can
            #  not automatically handle some DDP pre-hooks of `DistributedDataParallel`.
            out = cfg.rt_model(input_student)

            # Apply mutual distillation if required.
            # TODO: better handling `requires_grad` of the main `sup_loss` backward
            if cfg.msk_config.mutual_dis:
                sup_loss = (
                    cfg.msk_config.aux_loss_coef
                    * 0.5
                    * (
                        torch.nn.KLDivLoss(reduction="batchmean")(
                            F.log_softmax(out, dim=1),
                            F.softmax(soft_out.detach(), dim=1),
                        )
                        + torch.nn.KLDivLoss(reduction="batchmean")(
                            F.log_softmax(soft_out, dim=1),
                            F.softmax(out.detach(), dim=1),
                        )
                    )
                )
                sup_loss.backward()
            elif cfg.msk_config.decoupled_kl:
                ctx._mask_optimizer.store_grad(ctx._cur_subnet.idx, "sup_soft_grad")
                mask_to_backward = (
                    chosen_subnet.non_solidified_masks
                    if solidify and not cfg.msk_config.solidify_grad
                    else chosen_subnet.masks
                )
                if not cfg.msk_config.auto_t:
                    sup_loss = ctx._decoupled_kl_loss(out, soft_out, mask_to_backward)
                else:
                    normalized_shape = (out.size()[1],)
                    soft_out_auto_t = F.layer_norm(soft_out, normalized_shape, None, None, 1e-7)
                    out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                    sup_loss = ctx._decoupled_kl_loss(out_auto_t, soft_out_auto_t, mask_to_backward)
            else:
                if not cfg.msk_config.auto_t:
                    sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(out, dim=1), F.softmax(soft_out.detach(), dim=1)
                    )
                else:
                    normalized_shape = (out.size()[1],)
                    soft_out_auto_t = F.layer_norm(soft_out.detach(), normalized_shape, None, None, 1e-7)
                    out_auto_t = F.layer_norm(out, normalized_shape, None, None, 1e-7)
                    sup_loss = cfg.msk_config.aux_loss_coef * torch.nn.KLDivLoss(reduction="batchmean")(
                        F.log_softmax(out_auto_t, dim=1),
                        F.softmax(soft_out_auto_t.detach(), dim=1),
                    )
                sup_loss.backward()

            ctx._aux_losses.update(sup_loss.item(), images.size(0))
            cfg.rt_tmp_var.extra_log += f"Aux_Loss {ctx._aux_losses.val:.4f} ({ctx._aux_losses.avg:.4f})\t"

        if not cfg.msk_config.decoupled_kl:
            for msk in masks_list:
                msk.requires_grad = True
        del ctx._req_soft_forward
        del ctx._req_hard_forward

        # Update the mask if it is not hard forward and not all being solidified.
        # If not requiring hard forward and under `solidify` mode, when the `solidify_grad` is enabled,
        # the mask will be updated all the time.
        if (
            not cfg.msk_config.disable_msk_optim
            and not ctx._req_hard_forward
            and (not all_solidified or (solidify_grad and cfg.msk_config.update_after_solidify))
        ):
            optim_start_time = time.time()
            ctx._mask_optimizer.cus_step(ctx._cur_subnet.idx, "sub_optim", ctx._cur_subnet.mut_scale)
            ctx._timer["optim_time"].update(time.time() - optim_start_time)

        if cfg.msk_config.multi_target:
            # For multiple FLOPs targets, we should solidify all masks at the same time.
            if solidify:
                for mask_idx in range(len(ctx._pool)):
                    # noinspection PyUnboundLocalVariable
                    all_solidified = all([gp.solidified[mask_idx] for gp in groups])
                    if not all_solidified:
                        chosen_subnet = ctx._pool[mask_idx]
                        ctx._solidify(mask_idx, locals())
        else:
            if solidify and not all_solidified:
                ctx._solidify(mask_idx, locals())

    def _forward_impl_student(self: __DynamicResNet, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.layer1):
            if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw and i >= self._cur_depth[0]:
                break
            x = layer(x)

        for i, layer in enumerate(self.layer2):
            if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw and i >= self._cur_depth[1]:
                break
            x = layer(x)

        for i, layer in enumerate(self.layer3):
            if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw and i >= self._cur_depth[2]:
                break
            x = layer(x)

        for i, layer in enumerate(self.layer4):
            if cfg.msk_config.iterative_pruning and cfg.msk_config.ip_dw and i >= self._cur_depth[3]:
                break
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x


class BasicDepthWidthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "dw_basic"
    __DynamicResNet = "Union[BasicDepthWidthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        # Binary dw sampling by default.
        width_candid = [0.5, 1]
        depth_candid = [[1, _d] for _d in self.stage_list]
        width_product = itertools.product(width_candid, repeat=4)
        depth_product = itertools.product(*depth_candid)
        return list(itertools.product(depth_product, width_product))


class ManualDepthWidthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "dw_manual"
    __DynamicResNet = "Union[ManualDepthWidthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        return [
            [[1, 1, 1, 1], [0.3, 0.3, 0.3, 0.3]],
            [[2, 1, 3, 3], [0.3, 0.4, 0.4, 0.4]],
            [[3, 4, 6, 3], [1.0, 1.0, 1.0, 1.0]],
        ]


class AutoDepthWidthSpaceMixIn(SpaceMixIn):
    __proxy_space__ = "dw_auto"
    __DynamicResNet = "Union[AutoDepthWidthSpaceMixIn, DynamicResNetBase]"

    @lru_cache(maxsize=1)
    def get_all_subnets(self: __DynamicResNet, **kwargs):
        """
        Get the sampling space of subnet proxies.

        A subnet proxy is assumed to be [depth_proxy, width_proxy],
        where depth_proxy is a list of integers and width_proxy is a list of floats.

        The sampling space is a list of all possible subnet proxies.
        """

        """ Prepare for building sampling space """

        # Fetch some parameters from cfg and kwargs
        subnet_num = cfg.ast_config.subnet_num
        min_width = cfg.ast_config.min_width
        subnet_range = cfg.ast_config.subnet_range
        width_granularity = cfg.ast_config.width_granularity
        target_flops = cfg.ast_config.target_flops
        bin_count = cfg.ast_config.bin_count
        bin_flops_range = cfg.ast_config.bin_flops_range
        subnet_num_per_bin = cfg.ast_config.subnet_num_per_bin
        space_build_scheme = cfg.ast_config.space_build_scheme
        include_min_max = cfg.ast_config.include_min_max

        # Analyze the subnet_range
        if subnet_range == "auto":
            subnet_min_depth = [1] * len(self.stage_list)
            subnet_max_depth = self.stage_list
            subnet_min_width = [min_width] * len(self.stage_list)
            subnet_max_width = [1.0] * len(self.stage_list)
        else:
            try:
                (
                    [subnet_min_depth, subnet_max_depth],
                    [
                        subnet_min_width,
                        subnet_max_width,
                    ],
                ) = subnet_range
            except Exception:
                raise ValueError("Illegal subnet_range detected!")

        # Check consistency of min_depth, max_depth, min_width, max_width
        assert (
            len(subnet_min_depth)
            == len(subnet_max_depth)
            == len(subnet_min_width)
            == len(subnet_max_width)
            == len(self.stage_list)
        ), "The length of subnet_min_depth, subnet_max_depth, subnet_min_width, subnet_max_width should be 4"
        assert all([_min <= _max for _min, _max in zip(subnet_min_depth, subnet_max_depth)]), (
            "The subnet_min_depth should be less than subnet_max_depth"
        )
        assert all([_min <= _max for _min, _max in zip(subnet_min_width, subnet_max_width)]), (
            "The subnet_min_width should be less than subnet_max_width"
        )

        # Obtain all candidates for depth and width in each stage
        all_depth_candid = [list(range(_min, _max + 1)) for _min, _max in zip(subnet_min_depth, subnet_max_depth)]
        all_width_candid = [
            list(
                np.arange(_min, _max + width_granularity, width_granularity).round(
                    len(str(width_granularity).split(".")[1])
                )
            )
            for _min, _max in zip(subnet_min_width, subnet_max_width)
        ]

        # Derive `subnet_num_per_bin` if it is set to 'auto'
        if subnet_num_per_bin == "auto":
            # If the subnet_num cannot be divided by bin_count, round it up to an integer
            subnet_num_per_bin = [(subnet_num + bin_count - 1) // bin_count] * bin_count
        else:
            # Check the sum of `subnet_num_per_bin`
            assert sum(subnet_num_per_bin) == subnet_num, (
                "The sum of `subnet_num_per_bin` should be equal to `subnet_num`."
            )

        # Check `subnet_num_per_bin`
        assert (
            isinstance(subnet_num_per_bin, list)
            and all([isinstance(_, int) for _ in subnet_num_per_bin])
            and len(subnet_num_per_bin) == bin_count
        ), "The `subnet_num_per_bin` should be a list of int with length equal to bin_count."

        # Set up bin_flops_range automatically if it is not specified.
        # Firstly, uniformly divide the flops range into bin_count parts based on the min/max subnet FLOPs.
        # Then, insert the target_flops into the bin_flops_range.
        if bin_flops_range == "auto":
            min_subnet_flops = self._estimate_flops(
                [[_d[0] for _d in all_depth_candid], [_w[0] for _w in all_width_candid]]
            )
            max_subnet_flops = self._estimate_flops(
                [
                    [_d[-1] for _d in all_depth_candid],
                    [_w[-1] for _w in all_width_candid],
                ]
            )
            bin_flops_range = np.linspace(min_subnet_flops, max_subnet_flops, bin_count)
            bin_flops_range = list(
                np.insert(
                    bin_flops_range,
                    np.searchsorted(bin_flops_range, target_flops),
                    target_flops,
                )
            )

        # Check bin_flops_range
        assert isinstance(bin_flops_range, list) and len(bin_flops_range) == (bin_count + 1), (
            "The `bin_flops_range` should be a list with length of `bin_count + 1`"
        )

        # TODO: buggy when `bin_count` is too large, need to be fixed later
        """ 
        Build sampling space based on bins

        Input arguments:
            all_depth_candid: list of list, each element is a list of candidate depth for each stage
            all_width_candid: list of list, each element is a list of candidate width for each stage 
            bin_count: int, the number of bins
            bin_flops_range: list containing the boundary points of each bin
            subnet_num_per_bin: list, the number of subnets to be sampled in each bin
            others: see above
        """

        if space_build_scheme == "greedy_build":
            all_subnets, subnet_flops = zip(
                *sorted(
                    map(
                        lambda _sub: (_sub, self._estimate_flops(_sub)),
                        itertools.product(
                            itertools.product(*all_depth_candid),
                            itertools.product(*all_width_candid),
                        ),
                    ),
                    key=lambda x: x[1],
                )
            )

            # Turn all_subnets and subnet_flops into numpy array for fast indexing
            all_subnets = np.array(all_subnets)
            subnet_flops = np.array(subnet_flops)

            def _choose_idx(_bin_idx):
                try:
                    return np.random.choice(
                        np.arange(bin_idx_bound[_bin_idx], bin_idx_bound[_bin_idx + 1]),
                        subnet_num_per_bin[_bin_idx],
                        replace=False,
                    )
                except ValueError:
                    return np.array([], dtype=np.int64)

            bin_idx_bound = np.searchsorted(subnet_flops, bin_flops_range)
            bin_sample_indices = np.concatenate([_choose_idx(_i) for _i in range(bin_count)])

            # Derive sampling space based on bin_sample_indices. The sampling space should be like:
            # [(Bin_1) subnet_1, subnet_2, ..., subnet_n1, (Bin_2) subnet_n1+1, subnet_n1+2, ..., subnet_n1+n2, ...]
            sample_space = all_subnets[bin_sample_indices].tolist()
        else:
            raise NotImplementedError

        """ Other stuffs for post processing """

        if include_min_max:
            # The sample space will be revised to:
            # [min_subnet (Bin_1) subnet_1, subnet_2, ..., subnet_n1,
            # (Bin_2) subnet_n1+1, subnet_n1+2, ..., subnet_n1+n2, ..., max_subnet]
            min_subnet_proxy = [
                [_d[0] for _d in all_depth_candid],
                [_w[0] for _w in all_width_candid],
            ]
            max_subnet_proxy = [
                [_d[-1] for _d in all_depth_candid],
                [_w[-1] for _w in all_width_candid],
            ]
            if min_subnet_proxy not in sample_space:
                sample_space = [
                    min_subnet_proxy,
                ] + sample_space
            if max_subnet_proxy not in sample_space:
                sample_space = sample_space + [
                    max_subnet_proxy,
                ]

        # Type check for sample space, all depth_proxy should be a list of int,
        # and all width_proxy should be a list of float. If not satisfied, turn them into such type.
        sample_space = [[list(map(int, _d)), list(map(float, _w))] for _d, _w in sample_space]

        space_size = len(sample_space)
        if space_size != subnet_num:
            warnings.warn(
                f"The scale of the generated space ({space_size}) differs from the assigned value ({subnet_num}), "
                f"probably because of automatic rounding, min-max addition or bin discarding.",
                category=UserWarning,
                stacklevel=2,
            )

        return sample_space

    def _estimate_flops(
        self: __DynamicResNet,
        subnet_proxy: list[Union[list[int], list[float]]],
    ):
        return sum([_d * (_w**2) for _d, _w in zip(*subnet_proxy)]) / sum(self.stage_list)


class BasicDepthWidthProxyMixIn(BasicDepthProxyMixIn, BasicWidthProxyMixIn):
    __dimension__ = "dw_basic"
    __DynamicResNet = "Union[BasicDepthWidthProxyMixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        BasicDepthProxyMixIn._post_initialize(self)
        BasicWidthProxyMixIn._post_initialize(self)

    def _apply_proxy(self: __DynamicResNet, proxy):
        BasicDepthProxyMixIn._apply_proxy(self, proxy[0])
        BasicWidthProxyMixIn._apply_proxy(self, proxy[1])

    def _clear_proxy(self: __DynamicResNet):
        BasicDepthProxyMixIn._clear_proxy(self)
        BasicWidthProxyMixIn._clear_proxy(self)


class WarmUpL1IdxDepthMixIn(MixIn):
    __output_indices__ = "warm_up_l1_depth"
    __DynamicResNet = "Union[WarmUpL1IdxDepthMixIn, DynamicResNetBase]"

    class __Sequential(BasicDepthProxyMixIn.Sequential):
        _l1_order: Tensor

    def _post_initialize(self: __DynamicResNet):
        cfg.register_pre_batch_hook(self.__pre_batch_hook)

        cast(Type[BasicDepthProxyMixIn], self.__class__)._sequential_get_out_indices.register(
            WarmUpL1IdxDepthMixIn, self.__class__.__sequential_get_out_indices
        )

        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            layer.register_buffer("_l1_order", torch.arange(len(layer)), persistent=True)

    def __sequential_get_out_indices(
        ctx: __DynamicResNet,
        self: __Sequential,
        input: Tensor | TensorWithMeta,
        depth: int,
    ) -> SliceType:
        return self._l1_order[:depth]

    @exec_range(fn_range=lambda _ep: _ep == cfg.st_warmup_epochs, ep_only_once=True)
    def __pre_batch_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        for l_name, layer in zip(
            ("layer1", "layer2", "layer3", "layer4"),
            (ctx.layer1, ctx.layer2, ctx.layer3, ctx.layer4),
        ):
            layer = cast(ctx.__Sequential, layer)

            block_metrics = []
            for b_name, block in layer.named_children():
                assert isinstance(block, Bottleneck) or isinstance(block, BasicBlock)
                metrics_val = 0
                metrics_numel = 0
                for c_name, conv in filter(lambda _m: isinstance(_m[1], nn.Conv2d), block.named_children()):
                    metrics_val += conv.weight.detach().abs().sum()
                    metrics_numel += conv.weight.numel()
                block_metrics.append(torch.div(metrics_val, metrics_numel)[None])

            block_metrics = torch.concatenate(block_metrics)

            single_zero = torch.zeros_like(block_metrics[0:1])
            # The first block is always kept, so we set its order to 0.
            layer._l1_order = torch.cat([single_zero, (torch.argsort(block_metrics[1:], descending=True) + 1)]).long()


class WarmUpL1IdxWidthMixIn(MixIn):
    __output_indices__ = "warm_up_l1_width"
    __DynamicResNet = "Union[WarmUpL1IdxWidthMixIn, DynamicResNetBase]"

    class __Conv2d(BasicWidthProxyMixIn.Conv2d):
        _l1_order: Tensor

    def _post_initialize(self: __DynamicResNet):
        cfg.register_pre_batch_hook(self.__pre_batch_hook)

        cast(Type[BasicWidthProxyMixIn], self.__class__)._conv2d_get_out_indices.register(
            WarmUpL1IdxWidthMixIn, self.__class__.__conv2d_get_out_indices
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and n != "conv1":
                m.register_buffer("_l1_order", torch.arange(m.out_channels), persistent=True)

    def __conv2d_get_out_indices(
        ctx: __DynamicResNet,
        self: __Conv2d,
        input: Tensor,
        in_meta: ForwardMeta,
        new_out_num: int,
    ) -> SliceType:
        return self._l1_order[:new_out_num]

    @exec_range(fn_range=lambda _ep: _ep == cfg.st_warmup_epochs, ep_only_once=True)
    def __pre_batch_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        for conv in filter(lambda _m: isinstance(_m, BasicWidthProxyMixIn.Conv2d), ctx.modules()):
            conv = cast(ctx.__Conv2d, conv)
            weight_l1 = conv.weight.detach().abs().sum(dim=(1, 2, 3))
            conv._l1_order = torch.argsort(weight_l1, descending=True)


class WarmUpL1IdxWidthDepMixIn(WarmUpL1IdxWidthMixIn):
    __output_indices__ = "warm_up_l1_width_dep"
    __DynamicResNet = "Union[WarmUpL1IdxWidthDepMixIn, BasicWidthProxyMixIn, DynamicResNetBase]"

    class __Conv2d(BasicWidthProxyMixIn.Conv2d):
        _l1_order: Tensor

    @exec_range(fn_range=lambda _ep: _ep == cfg.st_warmup_epochs, ep_only_once=True)
    def __pre_batch_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        for _dep_group in ctx._dep_groups:
            _dep_group = cast(ctx.DependencyGroup, _dep_group)
            # Average over all `Conv2d` in the same group
            weight_l1 = torch.stack(
                [_m.weight.detach().abs().sum(dim=(1, 2, 3)) for _m in _dep_group.layers.values()],
                dim=0,
            ).mean(dim=0)
            l1_order = torch.argsort(weight_l1, descending=True)
            for _m in _dep_group.layers.values():
                _m = cast(ctx.__Conv2d, _m)
                _m._l1_order = l1_order

    _WarmUpL1IdxWidthMixIn__pre_batch_hook = __pre_batch_hook


class WarmUpL1IdxDepthWidthMixIn(WarmUpL1IdxDepthMixIn, WarmUpL1IdxWidthMixIn):
    __output_indices__ = "warm_up_l1"
    __DynamicResNet = "Union[WarmUpL1IdxDepthWidthMixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        WarmUpL1IdxDepthMixIn._post_initialize(self)
        WarmUpL1IdxWidthMixIn._post_initialize(self)


class WarmUpL1IdxDepthWidthDepMixIn(WarmUpL1IdxDepthMixIn, WarmUpL1IdxWidthDepMixIn):
    __output_indices__ = "warm_up_l1_dep"
    __DynamicResNet = "Union[WarmUpL1IdxDepthWidthDepMixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        WarmUpL1IdxDepthMixIn._post_initialize(self)
        WarmUpL1IdxWidthDepMixIn._post_initialize(self)


class WeightL1VizMixIn(MixIn):
    __st_track_l1__ = True
    __DynamicResNet = "Union[WeightL1VizMixIn, DynamicResNetBase]"

    class __Conv2d(nn.Conv2d):
        _weight_l1: float
        _weight_l1_selected: float
        _weight_l1_unselected: float

    def _post_initialize(self: __DynamicResNet):
        # Note that, we cannot register hooks to `Sequential`,
        # because its hooks will not be inherited by its slice.
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and n != "conv1":
                m.register_forward_hook(cast(Callable, partial(self.__conv2d_forward_hook, m_name=n)))
        cfg.register_post_student_hook(self.__post_student_hook)

    @DynamicResNetBase.student_only()
    def __conv2d_forward_hook(
        ctx: __DynamicResNet,
        module: __Conv2d,
        args: Tuple[TensorWithMeta | Tensor],
        output: TensorWithMeta | Tensor,
        m_name: str,
    ):
        if cfg.rt_mode != "train":
            return

        if isinstance(args[0], TensorWithMeta):
            out_indices = output.meta_data.out_indices
            in_indices = args[0].meta_data.out_indices
        else:
            out_indices = slice(None, module.out_channels)
            in_indices = slice(None, module.in_channels)

        # Track the L1 of the weights
        _weight = module.weight.detach()
        _weight_selected = _weight[out_indices][:, in_indices]
        _weight_l1 = _weight.abs().sum().item()
        _weight_l1_size = _weight.numel()
        _weight_l1_selected = _weight_selected.abs().sum().item()
        _weight_l1_selected_size = _weight_selected.numel()
        _weight_l1_unselected = _weight_l1 - _weight_l1_selected
        _weight_l1_unselected_size = _weight_l1_size - _weight_l1_selected_size

        # Smoothing buffer strategy
        if not hasattr(module, "_weight_l1"):
            module._weight_l1 = 0.0
            module._weight_l1_selected = 0.0
            module._weight_l1_unselected = 0.0

        if _weight_selected.shape[0] == module.out_channels:
            # Retrieve from the buffer
            log_l1 = module._weight_l1
            log_l1_selected = module._weight_l1_selected
            log_l1_unselected = module._weight_l1_unselected
        else:
            # Update the buffer
            module._weight_l1 = log_l1 = _weight_l1 / _weight_l1_size
            module._weight_l1_selected = log_l1_selected = _weight_l1_selected / _weight_l1_selected_size
            module._weight_l1_unselected = log_l1_unselected = _weight_l1_unselected / _weight_l1_unselected_size

        log_dict = {
            "weight_l1": log_l1,
            "weight_l1_selected": log_l1_selected,
            "weight_l1_unselected": log_l1_unselected,
        }
        cfg.rt_tb_logger.add_scalars(f"{m_name}/l1_value", log_dict, cfg.rt_iter)

    def __post_student_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        if cfg.rt_mode != "train":
            return

        for l_name, layer in zip(
            ("layer1", "layer2", "layer3", "layer4"),
            (ctx.layer1, ctx.layer2, ctx.layer3, ctx.layer4),
        ):
            block_metrics = {}
            for b_name, block in layer.named_children():
                assert isinstance(block, Bottleneck) or isinstance(block, BasicBlock)
                # Although it seems that a bug,
                # it is not that the temporary variable `_weight_l1_selected` in the block is not cleared.
                # It can ensure that when the block is not selected, its value is still smooth rather than 0.
                metrics = []
                for conv in filter(lambda _m: isinstance(_m, nn.Conv2d), block.children()):
                    metrics.append(getattr(conv, "_weight_l1_selected", 0.0))
                block_metrics[f"{b_name}.weight_l1_selected"] = sum(metrics) / len(metrics)

            cfg.rt_tb_logger.add_scalars(f"{l_name}/l1_value", block_metrics, cfg.rt_iter)


class WeightL1OrderVizMixIn(MixIn):
    __st_track_order__ = True
    __DynamicResNet = "Union[WeightL1OrderVizMixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        cfg.register_post_teacher_hook(self.__post_teacher_hook)

    def __post_teacher_hook(ctx: __DynamicResNet, g_dict: Dict, l_dict: Dict):
        for l_name, layer in zip(
            ("layer1", "layer2", "layer3", "layer4"),
            (ctx.layer1, ctx.layer2, ctx.layer3, ctx.layer4),
        ):
            layer = cast(nn.Sequential, layer)

            block_metrics = []
            for b_name, block in layer.named_children():
                assert isinstance(block, Bottleneck) or isinstance(block, BasicBlock)
                metrics_val = 0
                metrics_numel = 0
                for c_name, conv in filter(lambda _m: isinstance(_m[1], nn.Conv2d), block.named_children()):
                    conv = cast(nn.Conv2d, conv)
                    _weight = conv.weight.detach()
                    # Larger L1 is expected to be better
                    _weight_l1_order = _weight.abs().sum(dim=(1, 2, 3)).argsort(descending=True).tolist()
                    # Lower gradient sign is expected to be better
                    _weight_grad_sign_order = (conv.weight.grad * _weight.sign()).sum(dim=(1, 2, 3)).argsort().tolist()

                    # TODO: still not a good way to visualize the order
                    _log_name = ".".join((l_name, b_name, c_name))
                    cfg.rt_tb_logger.add_text(f"{_log_name}/l1_order", repr(_weight_l1_order), cfg.rt_iter)
                    cfg.rt_tb_logger.add_text(
                        f"{_log_name}/grad_order",
                        repr(_weight_grad_sign_order),
                        cfg.rt_iter,
                    )

                    metrics_val += conv.weight.detach().abs().sum()
                    metrics_numel += conv.weight.numel()
                block_metrics.append(torch.div(metrics_val, metrics_numel)[None])

            block_metrics = torch.concatenate(block_metrics)
            _layer_l1_order = torch.argsort(block_metrics, descending=True).tolist()

            cfg.rt_tb_logger.add_text(f"{l_name}/l1_order", repr(_layer_l1_order), cfg.rt_iter)


class SubnetPerfTrackMixIn(MixIn):
    __st_track_subnet_perf__ = True
    __DynamicResNet = "Union[SubnetPerfTrackMixIn, DynamicResNetBase]"

    def _post_initialize(self: __DynamicResNet):
        cfg.register_post_student_hook(self.__post_student_hook)

    @staticmethod
    def __post_student_hook(g_dict: Dict, l_dict: Dict):
        sup_loss: Tensor = l_dict["sup_loss"]
        criterion: nn.Module = l_dict["criterion"]
        out: Tensor = l_dict["out"]
        target: Tensor = l_dict["target"]
        chosen_subnet = l_dict["chosen_subnet"]

        tar_loss = criterion(out.detach(), target)

        # Track both the `sup_loss` and `tar_loss` of the chosen subnet
        tb_logger = cfg.rt_tb_logger
        tb_logger.add_scalar(f"subnet_sup_loss/{chosen_subnet}", sup_loss.item(), cfg.rt_iter)
        tb_logger.add_scalar(f"subnet_tar_loss/{chosen_subnet}", tar_loss.item(), cfg.rt_iter)


__all__ = ["resnet_34", "resnet_50"]


def resnet_34(**kwargs):
    return create_dynamic_resnet(**kwargs)(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=cfg.num_classes,
    )


def resnet_50(**kwargs):
    return create_dynamic_resnet(**kwargs)(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=cfg.num_classes,
    )


if __name__ == "__main__":
    pass
