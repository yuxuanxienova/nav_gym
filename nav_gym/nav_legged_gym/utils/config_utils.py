# python
from dataclasses import dataclass, field, Field
import importlib
from typing import Callable, Any, Dict, Mapping, Iterable
from copy import deepcopy

# List of all methods provided by sub-module.
__all__ = ["configclass", "update_class_from_dict", "class_to_dict"]


"""
Wrapper around dataclass.
"""


def __dataclass_transform__():
    """Add annotations decorator for PyLance."""
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):
    """Wrapper around `dataclass` functionality to add extra checks and utilities.

    As of Python3.8, the standard dataclasses have two main issues which makes them non-generic for configuration use-cases.
    These include:
        1. Requiring a type annotation for all its members.
        2. Requiring explicit usage of `field(default_factory=...)` to reinitialize mutable variables.

    This function wraps around `dataclass` utility to deal with the above two issues.

    Reference:
        https://docs.python.org/3/library/dataclasses.html#dataclasses.Field
    """
    # add type annotations
    _add_annotation_types(cls)
    # add field factory
    _process_mutable_types(cls)
    # copy mutable members
    setattr(cls, "__post_init__", _custom_post_init)
    # wrap around dataclass
    cls = dataclass(cls, **kwargs)
    # return wrapped class
    return cls


"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj: object) -> Dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj (object): An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Dict[str, Any]: Converted dictionary mapping.
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    else:
        obj_dict = obj.__dict__
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        if callable(value):
            data[key] = f"{value.__module__}:{value.__name__}"
        # check if attribute is a dictionary or a class
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # elif isinstance(value, dict):
        #     for dict_key, dict_value in value.items():
        #         if callable(dict_value):
        #             value[dict_key] = f"{dict_value.__module__}:{dict_value.__name__}"
        #     data[key] = value
        else:
            data[key] = value
    return data


def string_to_callable(str):
    try:
        mod_name, attr_name = str.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError("imported object is not callable")

    except Exception as e:
        raise ValueError(
            f"While updating the config from a dictionnary, could not interpret an entry as a callable object. The format should be 'module:attribute_name'\n.\
                            While processing {str}, received error:\n {e}"
        )


def update_class_from_dict(obj, data: Dict[str, Any], ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj (object): An instance of a class to convert.
        data (Dict[str, Any]): Input dictionary to update from.

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():
        key_ns = ns + "/" + key
        if hasattr(obj, key):  # FIXME doesn't work when obj is a dict setattr()
            obj_mem = getattr(obj, key)
            if isinstance(obj_mem, Mapping):
                # recursively call if it is a dictionary
                for k, v in obj_mem.items():
                    if callable(v):
                        value[k] = string_to_callable(value[k])
                setattr(obj, key, value)
            elif isinstance(value, Mapping):
                # recursively call if it is a dictionary
                update_class_from_dict(obj_mem, value, ns=key_ns)
            elif isinstance(value, Iterable) and not isinstance(value, str):
                # check length of value to be safe
                if len(obj_mem) != len(value) and obj_mem is not None:
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}. Expected: {len(obj_mem)}, Received: {len(value)}."
                    )
                else:
                    setattr(obj, key, value)
            elif callable(obj_mem):
                value = string_to_callable(value)
                setattr(obj, key, value)

            elif isinstance(value, type(obj_mem)):
                # check that they are type-safe
                setattr(obj, key, value)
            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}. Expected: {type(obj_mem)}, Received: {type(value)}."
                )
        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")


"""
Private helper functions.
"""


def _add_annotation_types(cls):
    """Add annotations to all elements in the dataclass.

    By definition in Python, a field is defined as a class variable that has a type annotation.

    In case type annotations are not provided, dataclass ignores those members when :func:`__dict__()` is called.
    This function adds these annotations to the class variable to prevent any issues in case the user forgets to
    specify the type annotation.

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos = (0.0, 0.0, 0.0)
           ^^
           If the function is NOT used, the following type-error is returned:
           TypeError: 'pos' is a field but has no type annotation
    """
    # Note: Do not change this line. `cls.__dict__.get("__annotations__", {})` is different from `cls.__annotations__` because of inheritance.
    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    # cls.__annotations__ = dict()
    for key in dir(cls):
        # skip dunder members
        if key.startswith("__"):
            continue
        # add type annotations for members that are not classes
        var = getattr(cls, key)
        if not isinstance(var, type):
            if key not in cls.__annotations__:
                cls.__annotations__[key] = type(var)


def _process_mutable_types(cls):
    """Initialize all mutable elements through :obj:`dataclasses.Field` to avoid unnecessary complaints.

    By default, dataclass requires usage of :obj:`field(default_factory=...)` to reinitialize mutable objects every time a new
    class instance is created. If a member has a mutable type and it is created without specifying the `field(default_factory=...)`,
    then Python throws an error requiring the usage of `default_factory`.

    Additionally, Python only explicitly checks for field specification when the type is a list, set or dict. This misses the
    use-case where the type is class itself. Thus, the code silently carries a bug with it which can lead to undesirable effects.

    This function deals with this issue

    This makes the following a feasible operation:

    @dataclass
    class State:
        pos: list = [0.0, 0.0, 0.0]
           ^^
           If the function is NOT used, the following value-error is returned:
           ValueError: mutable default <class 'list'> for field pos is not allowed: use default_factory
    """

    def _return_f(f: Any) -> Callable[[], Any]:
        """Returns default function for creating mutable/immutable variables."""

        def _wrap():
            if isinstance(f, Field):
                return f.default_factory
            else:
                return f

        return _wrap

    for key in dir(cls):
        # skip dunder members
        if key.startswith("__"):
            continue
        # define explicit field for data members
        f = getattr(cls, key)
        if not isinstance(f, type):
            f = field(default_factory=_return_f(f))
            setattr(cls, key, f)


def _custom_post_init(obj):
    """Deepcopy all elements to avoid shared memory issues for mutable objects in dataclasses initialization.

    This function is called explicitly instead of as a part of :func:`_process_mutable_types()` to prevent mapping
    proxy type i.e. a read only proxy for mapping objects. The error is thrown when using hierarchical data-classes
    for configuration.
    """
    for key in dir(obj):
        # skip dunder members
        if key.startswith("__"):
            continue
        # duplicate data members
        var = getattr(obj, key)
        if not callable(var):
            setattr(obj, key, deepcopy(var))