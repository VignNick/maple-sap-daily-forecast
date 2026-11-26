# utils/yaml_registry.py
from .YAML_factory import build, serialize

class YAMLRegistry:
    _registry = {}

    @classmethod
    def register(cls, name: str, target, builder, serializer):
        assert name not in cls._registry, f"YAMLRegistry: Name '{name}' is already registered."
        cls._registry[name] = {
            "target": target,
            "builder": builder,
            "serializer": serializer
        }

    @classmethod
    def exists(cls, name: str):
        return name in cls._registry

    @classmethod
    def get(cls, name: str):
        return cls._registry[name]
    
    @classmethod
    def construct(cls, name: str, kwargs):
        entry = cls.get(name)
        return entry["builder"](entry["target"], kwargs)
    
    @classmethod
    def serialize(cls, name: str, obj):
        entry = cls.get(name)
        return entry["serializer"](obj)



def yaml_serializable(name=None, builder_method=None, save_method=None):
    def wrapper(cls):
        key = name or cls.__name__
        builder = builder_method or build
        serializer = save_method or serialize
        YAMLRegistry.register(key, cls, builder, serializer)
        
        cls.__yaml_serializable__ = True
        cls.__yaml_name__ = key

        return cls
    return wrapper