from typing import Dict, Type
import torch
import torch.nn as nn


ACTIVATION_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_activation(name: str):
    """
    Decorator to register an activation module by name.
    """
    def decorator(cls: Type[nn.Module]):
        if name in ACTIVATION_REGISTRY:
            raise ValueError(f"Activation '{name}' already registered")
        ACTIVATION_REGISTRY[name] = cls
        return cls
    return decorator

@register_activation('ReLU')
class ReLU(nn.ReLU):
    pass

@register_activation('LeakyReLU')
class LeakyReLU(nn.LeakyReLU):
    pass

@register_activation('SiLU')
class SiLU(nn.SiLU):
    pass

@register_activation('GELU')
class GELU(nn.GELU):
    pass

@register_activation('Tanh')
class Tanh(nn.Tanh):
    pass

@register_activation('Sigmoid')
class Sigmoid(nn.Sigmoid):
    pass

@register_activation('ELU')
class ELU(nn.ELU):
    pass

@register_activation('SquarePlus')
class SquarePlus(nn.Module):
    """
    SquarePlus Activation Function: f(x) = 0.5 * (x + sqrt(x^2 + b))
    """
    def __init__(self, b: float = 4.0):
        super().__init__()
        self.b = b
        self.__kwargs__ = {'b': b}

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x * x + self.b))
    
@register_activation('PositiveClip')
class PositiveClip(nn.Module):
    '''
    Module pour contraindre les sorties à être positives en utilisant une opération de détachement.\n
    Ainsi, les gradients ne sont pas affectés par le clipping.
    '''
    def __init__(self):
        '''
        Module pour contraindre les sorties à être positives en utilisant une opération de détachement.\n
        Ainsi, les gradients ne sont pas affectés par le clipping.
        '''
        super(PositiveClip, self).__init__()
    def forward(self, y):
        y_clipped = torch.clamp(y, min=0.0)
        return y + (y_clipped - y).detach()