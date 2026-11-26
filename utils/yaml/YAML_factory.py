# utils/YAML/YAML_factory.py

def build(target, kwargs):
    return target(**kwargs)

def serialize(obj):
    if hasattr(obj, "__dict__"):
        return make_yaml_safe(obj.__dict__)
    else:
        raise NotImplementedError(f"Object of type {type(obj)} is not serializable")
    
def _resolve_qualname(module_name, qualname):
    import importlib

    obj = importlib.import_module(module_name)

    for part in qualname.split("."):
        obj = getattr(obj, part)

    return obj

def make_yaml_safe(obj):
    from .YAML_registry import YAMLRegistry
    import inspect
    import enum

    # ---- Is YAML-serializable via registry? ----
    if getattr(obj.__class__, "__yaml_serializable__", False) or YAMLRegistry.exists(obj.__class__.__name__):
        yaml_name = getattr(obj.__class__, "__yaml_name__", obj.__class__.__name__)
        return {
            "__type__": "yaml_serialize_object",
            "module": obj.__class__.__module__,
            "class": yaml_name,
            "kwargs": make_yaml_safe(YAMLRegistry.serialize(yaml_name, obj))
        }

    # ---- Dict ----
    if isinstance(obj, dict):
        return {k: make_yaml_safe(v) for k, v in obj.items()}

    # ---- List / Tuple ----
    if isinstance(obj, (list, tuple)):
        return [make_yaml_safe(v) for v in obj]
        
    # ---- Enum ----
    if isinstance(obj, enum.Enum):
        return {
            "__type__": "enum",
            "enum_module": obj.__class__.__module__,
            "enum_qualname": obj.__class__.__qualname__,
            "value": obj.name,
        }

    # ---- Callable (function or class) ----
    if inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
        return {
            "__type__": "callable",
            "kind": "function",
            "obj_module": obj.__module__,
            "obj_qualname": obj.__qualname__
        }

    if inspect.isclass(obj):
        return {
            "__type__": "callable",
            "kind": "class",
            "obj_module": obj.__module__,
            "obj_qualname": obj.__qualname__
        }
    
    # ---- Si possède une méthode to_dict, on l'utilise pour sérialiser l'objet ----
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
            cls = obj.__class__

            name = getattr(cls, "__yaml_name__", cls.__name__)

            return {
                "__type__": "dict_object",
                "class": name,
                "value": make_yaml_safe(obj.to_dict())
            }

    return obj


def restore_yaml_safe(obj):
    from .YAML_registry import YAMLRegistry

    # ---- Dict ----
    if isinstance(obj, dict):
        # ---- typed objects ----
        if "__type__" in obj:
            t = obj["__type__"]

            # ---- Is YAML-serializable via registry? ----
            if t == "yaml_serialize_object":
                kwargs = restore_yaml_safe(obj["kwargs"])
                return YAMLRegistry.construct(obj["class"], kwargs)

            # enum
            if t == "enum":
                enum_cls = _resolve_qualname(obj["enum_module"], obj["enum_qualname"])
                return enum_cls[obj["value"]]

            # callable
            if t == "callable":
                return _resolve_qualname(obj["obj_module"], obj["obj_qualname"])
            
            # dict_object
            if t == "dict_object":
                cls = YAMLRegistry.get(obj["class"])
                value = restore_yaml_safe(obj["value"])
                return cls.from_dict(value)

        # normal dict
        return {k: restore_yaml_safe(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, list):
        return [restore_yaml_safe(v) for v in obj]
    
    if isinstance(obj, tuple):
        return tuple(restore_yaml_safe(v) for v in obj)
    
    return obj