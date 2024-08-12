# mypy: allow-untyped-defs
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# NOTE: [auto-functionalizing custom ops]
# Users may wish to torch.compile custom ops that mutate their inputs.
# torch.compile will automatically support this op without anyone needing
# to provide a functionalization kernel for it. Here's how.
#
# Let's say we have a hypothetical mylib::sin_(Tensor(a!) x) -> ()
# op. First, when FakeTensor sees this op:
# - If the schema says it returns nothing, we can generate a trivial
#   FakeTensor rule for it (that returns nothing).
# - Otherwise, the user needs to provide a FakeTensor impl (fake impl)
#
# Next, when Python FunctionalTensor sees the op, it will functionalize
# it by emitting a call to an auto_functionalize(op, ["x"], {"x": ...})
# HOP and replacing the mutated inputs with corresponding outputs of this HOP.
# This HOP effectively runs the functional version of the op when
# called: it clones inputs that will be mutated, runs the op, and
# then returns (output, Tensors with the new values)


class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.

    Concretely, it looks at all the arguments that are mutable through
    _mutable_op's operator schema, clones those kwargs, runs
    `out = _mutable_op(**kwargs)` with the cloned values, and then returns the
    operator output concatenated with the cloned values that were mutated.

    We have some restrictions on `_mutable_op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.

    The reason why _mutable_op is prefixed with an
    underscore is to prevent collisions with kwarg names in **kwargs.
    """

    def __init__(self):
        super().__init__("auto_functionalized")

    def __call__(
        self_,  # noqa: B902
        _mutable_op: OpOverload,
        **kwargs: Any,
    ) -> Tuple[Any, Tuple[Tensor, ...]]:
        assert can_auto_functionalize(_mutable_op)
        assert isinstance(kwargs, dict)
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized = AutoFunctionalized()
auto_functionalized.__module__ = "torch.ops.higher_order"


def can_auto_functionalize(op: OperatorBase) -> bool:
    if not isinstance(op, OpOverload):
        return False

    if torch._library.utils.is_builtin(op):
        # We control the built-ins. These may (in rare cases)
        # do input metadata mutation (which we have banned on custom ops)
        return False
    schema = op._schema
    if not schema.is_mutable:
        return False
    schema = op._schema

    for arg in schema.arguments:
        if arg.alias_info is None:
            continue
        if not arg.alias_info.is_write:
            continue
        if type(arg.type) is torch.TensorType:
            continue
        if (
            type(arg.type) is torch.OptionalType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        if (
            type(arg.type) is torch.ListType
            and type(arg.type.getElementType()) is torch.TensorType
        ):
            continue
        # Not yet supported: other Tensor types. This includes things like
        # Tensor?[], Tensor[]?.
        return False

    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.NoneType):
        # Skip schema returns -> None
        return True
    # The returns must not alias anything
    for ret in schema.returns:
        if ret.alias_info is None and type(ret.type) is torch.TensorType:
            continue
        # Not yet supported: List[Tensor] return.
        return False
    if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Functionalize"):
        return False
    return True


@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: OpOverload,
    _only_clone_these_tensors: Optional[Tuple[str, ...]] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    _all_aliased = kwargs.pop("_all_aliased", [])
    kwargs.pop("_arg_to_aliased", {})
    new_kwargs = dict(**kwargs)
    result = []

    _mutable_args_names = get_mutable_arg_names(_mutable_op)

    for name in _mutable_args_names:
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            new_kwargs[name] = (
                [clone_preserve_strides(x) for x in kwargs[name]]
                if kwargs[name] is not None and isinstance(kwargs[name], list)
                else (
                    clone_preserve_strides(kwargs[name])
                    if kwargs[name] is not None
                    else None
                )
            )
        result.append(new_kwargs[name])

    out = _mutable_op(**new_kwargs)
    latest_value_for = {}
    for alias in _all_aliased:
        # Do i need to wrap this like what happen in clone_preserve_strides in some dispatch things idk what they are?
        # item should see the effects of all of the mutated args that it is an alias for
        alias_with_effects = alias
        for name in _mutable_args_names:
            if (
                _only_clone_these_tensors is not None
                and name not in _only_clone_these_tensors
            ):
                # if the argument is mutated in place, item would have already observed the effect.
                pass
            else:
                # observe the effect that happened on `name`
                mutation_source = new_kwargs[name]
                alias_with_effects = alias_with_effects.as_strided_scatter(
                    mutation_source,
                    mutation_source.size(),
                    mutation_source.stride(),
                    mutation_source.storage_offset(),
                )
        result.append(alias_with_effects)

    if isinstance(out, tuple):
        return (*out, *result)  # type: ignore[return-value]
    else:
        return (out, *result)  # type: ignore[return-value]


@auto_functionalized.py_impl(FakeTensorMode)
def auto_functionalized_fake(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with mode:
        result = auto_functionalized_dense(_mutable_op, **kwargs)
        return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Dict[str, Any],
) -> Tuple[Any, Tuple[Tensor, ...]]:
    with disable_proxy_modes_tracing():
        out = auto_functionalized(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized,
        (_mutable_op,),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


def get_mutable_arg_names(op: OpOverload) -> List[str]:
    """
    Returns the list of argument names that get mutated according to the
    schema.
    """
    mutable_args_names = [
        arg.name
        for arg in op._schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names


def do_auto_functionalize(
    op: OpOverload,
    storage_to_aliases: Dict[int, List[int]],
    tensors_look_up: Dict[int, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI()

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}
    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    if "self" in unwrapped_kwargs or "self_" in unwrapped_kwargs:
        warnings.warn(
            "Using `self` or `self_` as an argument in the definition of custom ops may lead to ambiguous parsing. "
            "Please consider using a different name for this argument to avoid potential issues."
        )
    # List of the name of args that get mutated (according to the schema)
    mutable_args_names = get_mutable_arg_names(op)

    arg_to_aliased = {}
    all_aliased_addresses = set()

    for arg in mutable_args_names:
        ls = list(storage_to_aliases[normalized_kwargs[arg]._typed_storage()._cdata])

        # remove self from aliases.
        ls.remove(normalized_kwargs[arg]._cdata)

        all_aliased_addresses |= set(ls)

        # get tensors from tensors addresses.
        arg_to_aliased[arg] = ctx.unwrap_tensors([tensors_look_up[item] for item in ls])

    # take the union of all the aliased items.
    for arg in mutable_args_names:
        all_aliased_addresses.discard(normalized_kwargs[arg]._cdata)

    all_aliased_original = [tensors_look_up[item] for item in all_aliased_addresses]
    all_aliased_unwrapped = ctx.unwrap_tensors(all_aliased_original)

    with ctx.redispatch_to_next():
        # output of auto_functionalized is going to be as the following:
        # op_output, mutated_arg1, ..., all_aliased[0], ...
        unwrapped_outs = auto_functionalized(
            op, **dict(unwrapped_kwargs, _all_aliased=all_aliased_unwrapped, _arg_to_aliased=arg_to_aliased)  # type: ignore[arg-type]
        )

    unwrapped_actual_out: Union[Any, Tuple[Any]] = unwrapped_outs[
        : -len(mutable_args_names)
    ]

    unwrapped_mutable_out = unwrapped_outs[
        -(len(mutable_args_names) + len(all_aliased_original)) :
    ]

    if len(op._schema.returns) == 0:
        assert unwrapped_actual_out[0] is None
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        assert len(unwrapped_actual_out) == 1
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        assert len(unwrapped_actual_out) == len(op._schema.returns)

    original_args = [
        normalized_kwargs[name] for name in mutable_args_names
    ] + all_aliased_original

    for orig_arg, unwrapped_out in zip(original_args, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # We only handle Tensor or List[Tensor] here for now.
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        if isinstance(unwrapped_out, torch.Tensor):
            sync_update(unwrapped_out, orig_arg)
        elif isinstance(unwrapped_out, list) and all(
            isinstance(o, torch.Tensor) for o in unwrapped_out
        ):
            assert len(orig_arg) == len(unwrapped_out)
            for orig_a, o in zip(orig_arg, unwrapped_out):
                sync_update(o, orig_a)
        else:
            raise RuntimeError(
                f"unsupported type for auto-functionalization: {unwrapped_out}"
            )

    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]


@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)
