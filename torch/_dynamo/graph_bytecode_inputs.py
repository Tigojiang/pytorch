import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import local
from typing import Any, Protocol

import torch
from torch._dynamo.source import Source


PyCodegen = Any
UserObjectTrackingState = tuple[
    tuple[Callable[[PyCodegen], None], ...],
    tuple[weakref.ReferenceType[object], ...],
    tuple[tuple[str, int | None, int], ...],
    tuple[object, ...],
]


class _Device(Protocol):
    type: str
    index: int | None


# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution


# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs
class _RegistryState(local):
    def __init__(self) -> None:
        self.bytecode_constructors: list[Callable[[PyCodegen], None]] = []
        self.external_object_weakrefs: list[weakref.ReferenceType[object]] = []
        self.current_stream_indices: dict[tuple[str, int | None], int] = {}
        self.keep_alive: list[object] = []


_registry = _RegistryState()


def snapshot_bytecode_constructors() -> tuple[Callable[[PyCodegen], None], ...]:
    return tuple(_registry.bytecode_constructors)


def set_external_object_by_index(index: int, value: object) -> None:
    """Add or update an entry in the external object registry at runtime."""
    _registry.keep_alive.append(value)
    weakrefs = _registry.external_object_weakrefs
    if index == len(weakrefs):
        weakrefs.append(weakref.ref(value))
    elif index < len(weakrefs):
        weakrefs[index] = weakref.ref(value)
    else:
        raise AssertionError("Index past the end of index_to_user_object_weakref")


def get_external_object_by_index(index: int) -> object:
    weakrefs = _registry.external_object_weakrefs
    if index >= len(weakrefs):
        raise AssertionError("Index not registered in index_to_user_object_weakref")
    obj = weakrefs[index]()
    if obj is None:
        raise AssertionError("User object is no longer alive")
    return obj


def store_user_object_weakrefs(*args: object) -> None:
    _registry.external_object_weakrefs[:] = map(weakref.ref, args)


def call_with_external_object_state(
    fn: Callable[..., Any],
    current_stream_indices: tuple[tuple[str, int | None, int], ...],
    external_object_count: int,
    *args: object,
) -> Any:
    with restore_external_object_state(
        tuple(map(weakref.ref, args[:external_object_count])),
        {},
        current_stream_indices,
    ):
        return fn(*args[external_object_count:])


def wrap_with_additional_external_object_state(
    fn: Callable[..., Any], prefix_count: int
) -> Callable[..., Any]:
    weakrefs, current_stream_indices = snapshot_external_object_state()
    suffix = weakrefs[prefix_count:]
    live_objects = {
        prefix_count + offset: obj
        for offset, ref in enumerate(suffix)
        if (obj := ref()) is not None
    }

    def wrapped(*args: object, **kwargs: object) -> Any:
        prefix = snapshot_external_object_state()[0][:prefix_count]
        runtime_objects = live_objects.copy()
        for device_type, device_index, index in current_stream_indices:
            if index < prefix_count:
                continue
            device = torch.device(device_type, device_index)
            runtime_objects[index] = torch.accelerator.current_stream(device)
        with restore_external_object_state(
            prefix + suffix, runtime_objects, current_stream_indices
        ):
            return fn(*args, **kwargs)

    return wrapped


def snapshot_current_stream_indices() -> tuple[tuple[str, int | None, int], ...]:
    return tuple(
        (device_type, device_index, index)
        for (
            device_type,
            device_index,
        ), index in _registry.current_stream_indices.items()
    )


def snapshot_external_object_state() -> tuple[
    tuple[weakref.ReferenceType[object], ...],
    tuple[tuple[str, int | None, int], ...],
]:
    return tuple(_registry.external_object_weakrefs), snapshot_current_stream_indices()


def store_current_stream_indices(
    current_stream_indices: tuple[tuple[str, int | None, int], ...],
) -> None:
    _registry.current_stream_indices.clear()
    _registry.current_stream_indices.update(
        {
            (device_type, device_index): index
            for device_type, device_index, index in current_stream_indices
        }
    )


@contextmanager
def restore_external_object_state(
    weakrefs: tuple[weakref.ReferenceType[object], ...],
    live_objects: dict[int, object],
    current_stream_indices: tuple[tuple[str, int | None, int], ...],
) -> Iterator[None]:
    previous_weakrefs, previous_stream_indices = snapshot_external_object_state()
    previous_constructor_count = len(_registry.bytecode_constructors)
    previous_keep_alive_count = len(_registry.keep_alive)
    try:
        _registry.external_object_weakrefs[:] = weakrefs
        for index, obj in live_objects.items():
            _registry.external_object_weakrefs[index] = weakref.ref(obj)
        store_current_stream_indices(current_stream_indices)
        yield
    finally:
        _registry.external_object_weakrefs[:] = previous_weakrefs
        store_current_stream_indices(previous_stream_indices)
        del _registry.bytecode_constructors[previous_constructor_count:]
        del _registry.keep_alive[previous_keep_alive_count:]


def reset_user_object_tracking() -> None:
    _registry.bytecode_constructors.clear()
    _registry.external_object_weakrefs.clear()
    _registry.current_stream_indices.clear()
    _registry.keep_alive.clear()


def save_and_reset_user_object_tracking() -> UserObjectTrackingState:
    state = (
        tuple(_registry.bytecode_constructors),
        tuple(_registry.external_object_weakrefs),
        snapshot_current_stream_indices(),
        tuple(_registry.keep_alive),
    )
    reset_user_object_tracking()
    return state


def restore_user_object_tracking(state: UserObjectTrackingState) -> None:
    constructors, weakrefs, current_stream_indices, keep_alive = state
    _registry.bytecode_constructors[:] = constructors
    _registry.external_object_weakrefs[:] = weakrefs
    store_current_stream_indices(current_stream_indices)
    _registry.keep_alive[:] = keep_alive


def register_current_stream(device: _Device, index: int) -> None:
    _registry.current_stream_indices[(device.type, device.index)] = index


def get_current_stream_index(device: _Device) -> int | None:
    index = _registry.current_stream_indices.get((device.type, device.index))
    if index is None and device.index is not None:
        accelerator = torch.accelerator.current_accelerator()
        if (
            accelerator is not None
            and accelerator.type == device.type
            and device.index == torch.accelerator.current_device_index()
        ):
            index = _registry.current_stream_indices.get((device.type, None))
    return index


def register_graph_created_object(
    example_value: object, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    _registry.keep_alive.append(example_value)
    constructors = _registry.bytecode_constructors
    index = len(constructors)
    constructors.append(lambda cg: construct_fn(index, cg))
    try:
        _registry.external_object_weakrefs.append(weakref.ref(example_value))
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a user object to be used in the graph
def register_user_object(value: object, source: Source) -> int:
    constructors = _registry.bytecode_constructors
    index = len(constructors)
    constructors.append(lambda cg: cg(source))
    try:
        _registry.external_object_weakrefs.append(weakref.ref(value))
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a callback so invoke_leaf_function can retrieve nn.Module instances at runtime.
# We use a callback pattern instead of having invoke_leaf_function import get_external_object_by_index
# directly, because higher-order ops should not depend on dynamo (dynamo depends on them, not vice versa).
from torch._higher_order_ops.invoke_leaf_function import (
    set_leaf_function_module_retriever,
)


set_leaf_function_module_retriever(get_external_object_by_index)
