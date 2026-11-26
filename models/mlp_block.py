import torch.nn as nn
from .model import register_model, BaseModel
import models.activations as act_f
from utils.yaml.YAML_registry import yaml_serializable

@register_model('MLPBlock')
class MLPBlock(BaseModel):
    '''
    Multi-Layer Perceptron (MLP) block avec des couches séquentielles configurables.\n
    Chaque couche peut avoir sa propre dimension, normalisation, activation et dropout.\n
    Each layer can have its own dimension, normalization, activation, and dropout.
    '''
    @yaml_serializable(builder_method=lambda target, kwargs: target.from_dict(kwargs), save_method=lambda obj: obj.to_dict())
    class SequentialBlock():
        '''
        Configuration pour une couche dans le MLPBlock.
        '''
        def __init__(
                self,
                layer_dim: int,
                layer_norm: bool = False,
                activation: nn.Module | None = None,
                dropout_rate: float | None = None
            ):
            '''
            Definition de la configuration pour une couche dans le MLPBlock.
            Args:
                layer_dim (int ): Dimension de la sortie de lacouche.
                layer_norm (bool): Si True, applique LayerNorm après la couche linéaire.
                activation (nn.Module | None): Fonction d'activation à appliquer après la couche linéaire (et LayerNorm si applicable).\n
                    Si None, aucune activation n'est appliquée.
                dropout_rate (float | None): Taux de dropout à appliquer après l'activation.\n
                    Si None, aucun dropout n'est appliqué.
            '''
            self.layer_dim = layer_dim
            self.layer_norm = layer_norm
            self.activation = activation
            self.dropout_rate = dropout_rate

        @classmethod
        def from_dict(cls, config_dict: dict) -> 'MLPBlock.SequentialBlock':
            '''
            Crée une instance de SequentialBlock à partir d'un dictionnaire de configuration.
            Args:
                config_dict (dict): Dictionnaire contenant les paramètres de configuration.
            Returns:
                MLPBlock.SequentialBlock: Instance configurée.
            '''
            activation = None
            act_name = config_dict.get('activation', None)
            act_kwargs = config_dict.get('activation_kwargs', {})

            if act_name is not None:
                assert act_name in act_f.ACTIVATION_REGISTRY, f"Activation '{act_name}' non reconnue dans le registre de function d'activation."
                activation = act_f.ACTIVATION_REGISTRY[act_name](**act_kwargs)

            return cls(
                layer_dim=config_dict['layer_dim'],
                layer_norm=config_dict.get('layer_norm', False),
                activation=activation,
                dropout_rate=config_dict.get('dropout_rate', None)
            )

        def to_dict(self) -> dict:
            '''
            Sérialise la configuration du bloc séquentiel en un dictionnaire.
            Returns:
                dict: Dictionnaire contenant les paramètres de configuration.
            '''
            return {
                'layer_dim': self.layer_dim,
                'layer_norm': self.layer_norm,
                'activation': (
                    self.activation.__class__.__name__
                    if self.activation is not None
                    else None
                ),
                'activation_kwargs': (
                    self.activation.__kwargs__
                    if self.activation is not None and hasattr(self.activation, '__kwargs__')
                    else {}
                ),
                'dropout_rate': self.dropout_rate
            }

    def __init__(self, input_dim: int, output_dim: int, mlp_config: list[SequentialBlock] | None = None):
        '''
        Initialise le MLPBlock avec une configuration séquentielle.
        Args:
            input_dim (int): Dimension de l'entrée du MLPBlock. Écrase la dimension de la première couche dans mlp_config.
            output_dim (int): Dimension de la sortie du MLPBlock. Utiliser pour ajouter une couche linéaire finale.
            mlp_config (list[SequentialBlock] | None): Liste de configurations pour chaque couche dans le MLPBlock. Si None, utilise une configuration par défaut avec une seule couche de input_dim -> output_dim.
        '''
        super(MLPBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_config = mlp_config

        if mlp_config is None:
            self.network = nn.Linear(input_dim, output_dim)
            return
        
        layers = []

        for i, layer_conf in enumerate(mlp_config):
            layers.append(nn.Linear(input_dim if i == 0 else mlp_config[i-1].layer_dim, layer_conf.layer_dim))
            if layer_conf.layer_norm:
                layers.append(nn.LayerNorm(layer_conf.layer_dim))
            if layer_conf.activation is not None:
                layers.append(layer_conf.activation)
            if layer_conf.dropout_rate is not None:
                layers.append(nn.Dropout(layer_conf.dropout_rate))

        layers.append(nn.Linear(mlp_config[-1].layer_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du MLPBlock en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'mlp_config': [layer.to_dict() for layer in self.mlp_config] if self.mlp_config is not None else None
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MLPBlock':
        '''
        Crée une instance de MLPBlock à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            MLPBlock: Instance configurée.
        '''
        mlp_config = cls.from_dict_get_mlp_config(config_dict)
        
        return cls(
            input_dim=config_dict['input_dim'],
            output_dim=config_dict['output_dim'],
            mlp_config=mlp_config
        )
    
    @classmethod
    def from_dict_get_mlp_config(cls, config_dict: dict) -> list[SequentialBlock] | None:
        '''
        Extrait uniquement la configuration du MLP à partir d'un dictionnaire de configuration complet.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            list[SequentialBlock] | None: Liste de configurations pour chaque couche dans le MLPBlock, ou None si aucune configuration n'est trouvée.
        '''
        mlp_config = config_dict.get('mlp_config', None)
        if mlp_config is not None:
            mlp_config = mlp_config.get('mlp_config', mlp_config)
            mlp_config = [cls.SequentialBlock.from_dict(layer_conf) for layer_conf in mlp_config]
        
        return mlp_config