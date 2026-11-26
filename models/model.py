from typing import Dict, Type
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str):
    """
    Decorateur pour enregistrer un module de modèle par nom.
    """
    def decorator(cls: Type[nn.Module]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method must be implemented by subclasses")
    
    def to_dict(self) -> dict:
        """
        Sérialise la configuration du modèle en un dictionnaire.
        Doit être implémenté par les sous-classes.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration du modèle.
        """
        raise NotImplementedError("serialize_config must be implemented by subclasses")

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BaseModel':
        """
        Crée une instance du modèle à partir d'un dictionnaire de configuration.
        Doit être implémenté par les sous-classes.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration du modèle.
        Returns:
            BaseModel: Instance du modèle configurée selon le dictionnaire.
        """
        raise NotImplementedError("from_dict must be implemented by subclasses")