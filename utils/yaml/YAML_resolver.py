import enum
from .YAML_factory import make_yaml_safe, restore_yaml_safe


class YAMLResolver:
    """
    A class to resolve YAML tags and construct Python objects from them.
    """

    def __init__(self):
        pass

    @staticmethod
    def safe_save_yaml(data, file_path):
        import yaml, os
        if not file_path.endswith('.yaml'):
            file_path += '.yaml'
        if os.path.dirname(file_path) != '':
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.safe_dump(make_yaml_safe(data), f, sort_keys=False)

    @staticmethod
    def safe_load_yaml(file_path):
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return restore_yaml_safe(data)
    
    @staticmethod
    def _resolve_qualname(module_name, qualname):
        import importlib

        obj = importlib.import_module(module_name)

        for part in qualname.split("."):
            obj = getattr(obj, part)

        return obj

    # @staticmethod
    # def make_yaml_safe(obj):
    #     import inspect
    #     import torch
    #     from pathlib import Path

    #     # ---- Is YAML-serializable via registry? ----
    #     if getattr(obj.__class__, "__yaml_serializable__", False):
    #         print(f"Serializing object of type {type(obj)} using custom YAML serialization.")
    #         return {
    #             "__type__": "yaml_serialize_object",
    #             "class": getattr(obj.__class__, "__yaml_name__", obj.__class__.__name__),
    #             "value": YAMLResolver.make_yaml_safe(obj.to_dict())
    #         }

    #     # ---- Dict ----
    #     if isinstance(obj, dict):
    #         return {k: YAMLResolver.make_yaml_safe(v) for k, v in obj.items()}

    #     # ---- List / Tuple ----
    #     if isinstance(obj, (list, tuple)):
    #         return [YAMLResolver.make_yaml_safe(v) for v in obj]
        
    #     # ---- Enum ----
    #     if isinstance(obj, enum.Enum):
    #         return {
    #             "__type__": "enum",
    #             "enum_module": obj.__class__.__module__,
    #             "enum_qualname": obj.__class__.__qualname__,
    #             "value": obj.name,
    #         }

    #     # ---- Torch Dtype ----
    #     if isinstance(obj, torch.dtype):
    #         return {
    #             "__type__": "torch.dtype",
    #             "type": obj.__str__(),
    #             "value": str(obj)
    #         }

    #     # ---- Torch Device ----
    #     if isinstance(obj, torch.device):
    #         return {
    #             "__type__": "torch.device",
    #             "value": str(obj)
    #         }

    #     # ---- Pathlib ----
    #     if isinstance(obj, Path):
    #         return {
    #             "__type__": "path",
    #             "value": str(obj)
    #         }

    #     # ---- Callable (function or class) ----
    #     if inspect.isfunction(obj)or inspect.ismethod(obj) or inspect.isbuiltin(obj):
    #         return {
    #             "__type__": "callable",
    #             "kind": "function",
    #             "obj_module": obj.__module__,
    #             "obj_qualname": obj.__qualname__
    #         }

    #     if inspect.isclass(obj):
    #         return {
    #             "__type__": "callable",
    #             "kind": "class",
    #             "obj_module": obj.__module__,
    #             "obj_qualname": obj.__qualname__
    #         }

    #     # ---- registry-based object ----
    #     if hasattr(obj, "to_dict") and callable(obj.to_dict):
    #         cls = obj.__class__

    #         name = getattr(cls, "__yaml_name__", cls.__name__)

    #         return {
    #             "__type__": "object",
    #             "class": name,
    #             "value": YAMLResolver.make_yaml_safe(obj.to_dict())
    #         }

    #     print(f"Warning: Object of type {type(obj)} is not YAML serializable, returning as-is.")
    #     return obj

    # @staticmethod
    # def restore_yaml_safe(obj):
    #     import torch
    #     from pathlib import Path

    #     # ---- Dict ----
    #     if isinstance(obj, dict):
    #         # ---- typed objects ----
    #         if "__type__" in obj:
    #             t = obj["__type__"]

    #             # enum
    #             if t == "enum":
    #                 # import importlib
    #                 # module_name = obj["enum_module"]
    #                 # attr_name = obj["enum_qualname"]
    #                 # module = importlib.import_module(module_name)
    #                 enum_cls = YAMLResolver._resolve_qualname(obj["enum_module"], obj["enum_qualname"])
    #                 return enum_cls[obj["value"]]
                    

    #             # torch dtype
    #             if t == "torch.dtype":
    #                 return getattr(torch, obj["value"].split(".")[-1])

    #             # torch device
    #             if t == "torch.device":
    #                 return torch.device(obj["value"])

    #             # path
    #             if t == "path":
    #                 return Path(obj["value"])

    #             # callable
    #             if t == "callable":
    #                 # import importlib
    #                 # module_name = obj["obj_module"]
    #                 # attr_name = obj["obj_qualname"]
    #                 # obj = importlib.import_module(module_name)
    #                 # for part in attr_name.split("."):
    #                 #     obj = getattr(obj, part)

    #                 return YAMLResolver._resolve_qualname(obj["obj_module"], obj["obj_qualname"])
    #                 return getattr(module, attr_name)

    #             # ---- registry-based object (MAIN FEATURE) ----
    #             if t == "object":
    #                 # cls = YAMLRegistry.get(obj["class"])
    #                 # value = YAMLResolver.restore_yaml_safe(obj["value"])
    #                 # return cls.from_dict(value)
    #                 return 1

    #         # normal dict
    #         return {k: YAMLResolver.restore_yaml_safe(v) for k, v in obj.items()}

    #     # list / tuple
    #     if isinstance(obj, list):
    #         return [YAMLResolver.restore_yaml_safe(v) for v in obj]

    #     if isinstance(obj, tuple):
    #         return tuple(YAMLResolver.restore_yaml_safe(v) for v in obj)

    #     return obj