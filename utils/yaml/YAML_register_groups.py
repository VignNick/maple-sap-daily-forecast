# utils/yaml/YAML_regiter_group.py

from .YAML_registry import YAMLRegistry
from .YAML_factory import build

register_torch_done = False
def register_torch():
    import torch

    if hasattr(register_torch, "register_torch_done"):
        return

    # ---- Activations ----
    YAMLRegistry.register("ReLU", torch.nn.ReLU, build, lambda obj: {"inplace": obj.inplace})
    YAMLRegistry.register("LeakyReLU", torch.nn.LeakyReLU, build, lambda obj: {"inplace": obj.inplace, "negative_slope": obj.negative_slope})
    YAMLRegistry.register("SiLU", torch.nn.SiLU, build, lambda obj: {"inplace": obj.inplace})
    YAMLRegistry.register("GELU", torch.nn.GELU, build, lambda obj: {"approximate": obj.approximate})
    YAMLRegistry.register("Tanh", torch.nn.Tanh, build, lambda obj: {})
    YAMLRegistry.register("Sigmoid", torch.nn.Sigmoid, build, lambda obj: {})
    YAMLRegistry.register("ELU", torch.nn.ELU, build, lambda obj: {"inplace": obj.inplace, "alpha": obj.alpha})

    # ---- Devices ----
    YAMLRegistry.register(
        "device",
        torch.device,
        build,
        lambda obj: {"type": obj.type, "index": obj.index}
    )

    # ---- Dtypes ----
    YAMLRegistry.register(
        "dtype",
        torch.dtype,
        lambda target, kwargs: getattr(torch, kwargs["type"].split(".")[-1]),
        lambda obj: {"type": obj.__str__()}
    )

    register_torch.register_torch_done = True