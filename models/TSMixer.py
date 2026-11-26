from enum import Enum
import torch
import torch.nn as nn
import models.activations as act_f
from .model import register_model, BaseModel

class NormPlacement(Enum):
    PRE = "pre"
    POST = "post"
    NONE = "none"
    
@register_model('TSM_TemporalProjection')
class TSM_TemporalProjection(BaseModel):
    '''
    Projection temporelle pour aligner les dimensions temporelles d'entrée et de sortie.
    [Batch, Time_in (L), Channel] -> [Batch, Time_out (T), Channel]

    Partage des poids entre les différent channels, meme projection temporelle pour tous les channels.
    '''
    def __init__(self, input_dim, output_dim):
        '''
        [Batch, Time_in (L), Channel] -> [Batch, Time_out (T), Channel]

        Args:
            input_dim (int): Dimension temporelle d'entrée (L).
            output_dim (int): Dimension temporelle de sortie (T).
        '''
        super(TSM_TemporalProjection, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [Batch, Time_in (L), Channel]
        Returns:
            torch.Tensor: [Batch, Time_out (T), Channel]
        '''

        x = x.permute(0, 2, 1)  # [Batch, Channel, Time_in (L)]
        x = self.linear(x)       # [Batch, Channel, Time_out (T)]
        x = x.permute(0, 2, 1)  # [Batch, Time_out (T), Channel]

        return x
    
class TSM_MixingConfig:
    '''
    Configuration pour les modules de mélange temporel et de mélange de caractéristiques du TSMixer.
    Args:
        time_dim (int): Dimension temporelle d'entrée et de sortie du TSM_MixingLayer.
        channel_dim (int): Dimension des canaux d'entrée et de sortie du TSM_MixingLayer.
        hidden_dim (int): Dimension de la couche cachée dans le mélange de caractéristiques.\n
            Default: 64
        output_dim (int | None): Dimension 'feature' de sortie du mélange temporel.\n
            Si None, output_dim = channel_dim.
        norm_placement (NormPlacement): Placement de la normalisation.\n
            PRE: 2d batch normalisation avant l'activation (comme dans le TSMixer original)\n
            POST: 2d layer normalisation après l'activation\n
            NONE: pas de normalisation
        activation (nn.Module): Fonction d'activation.
        dropout_rate (float | None): Taux de dropout.
    '''
    def __init__(self,
                 time_dim: int,
                 channel_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int | None = None,
                 norm_placement: NormPlacement = NormPlacement.PRE,
                 activation:nn.Module = nn.ReLU(),
                 dropout_rate: float | None = None
                 ):
        '''
        Args:
            time_dim (int): Dimension temporelle d'entrée et de sortie du TSM_MixingLayer.
            channel_dim (int): Dimension des canaux d'entrée et de sortie du TSM_MixingLayer.
            hidden_dim (int): Dimension de la couche cachée dans le mélange de caractéristiques.\n
                Default: 64
            output_dim (int | None): Dimension 'feature' de sortie du mélange temporel.\n
                Si None, output_dim = channel_dim.
            norm_placement (NormPlacement): Placement de la normalisation.\n
                PRE: 2d batch normalisation avant l'activation (comme dans le TSMixer original)\n
                POST: 2d layer normalisation après l'activation\n
                NONE: pas de normalisation\n
            activation (nn.Module): Fonction d'activation.
            dropout_rate (float | None): Taux de dropout.
        '''
        self.time_dim = time_dim
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else channel_dim
        self.norm_placement = norm_placement
        self.activation = activation
        self.dropout_rate = dropout_rate

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TSMixerExt_Config':
        '''
        Crée une instance de TSMixerExt_Config à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            TSMixerExt_Config: Instance configurée.
        '''
        norm_placement = NormPlacement(config_dict['norm_placement'])
        
        activation = None
        act_name = config_dict.get('activation', None)
        act_kwargs = config_dict.get('activation_kwargs', {})
        if act_name is not None:
            assert act_name in act_f.ACTIVATION_REGISTRY, f"Activation '{act_name}' non reconnue dans le registre de function d'activation."
            activation = act_f.ACTIVATION_REGISTRY[act_name](**act_kwargs)

        return cls(
            time_dim=config_dict['time_dim'],
            channel_dim=config_dict['channel_dim'],
            hidden_dim=config_dict['hidden_dim'],
            output_dim=config_dict['output_dim'],
            norm_placement=norm_placement,
            activation=activation,
            dropout_rate=config_dict.get('dropout_rate', None)
        )
    
    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du TSMixer en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return {
            'time_dim': self.time_dim,
            'channel_dim': self.channel_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'norm_placement': self.norm_placement.value,
            'activation': self.activation.__class__.__name__,
            'activation_kwargs': (
                    self.activation.__kwargs__
                    if self.activation is not None and hasattr(self.activation, '__kwargs__')
                    else {}
                ),
            'dropout_rate': self.dropout_rate
        }

class PreBatchNormTime(nn.Module):
    """Pre-normalization for TSMixer (2D BatchNorm equivalent)."""
    def __init__(self, channel_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(channel_dim)

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)   # [C, B, L]
        x = self.bn(x)            # normalize over [B, L] per channel
        x = x.permute(0, 2, 1)    # [B, L, C] back
        return x

@register_model('TSM_TimeMixing')
class TSM_TimeMixing(BaseModel):
    '''
    Module de mélange temporel pour combiner les informations temporelles à travers les différentes étapes du TSMixer.
    [Batch, Time (T), Channel] -> [Batch, Time (T), Channel]
    '''
    def __init__(self, config: TSM_MixingConfig):
        super(TSM_TimeMixing, self).__init__()
        self.config = config

        self.fc = nn.Linear(config.time_dim, config.time_dim)
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate is not None else nn.Identity()

        # --- Selectionner la function 'forward' en fonction du placement de la normalisation ---
        if config.norm_placement == NormPlacement.PRE:
            self._forward = self._forward_pre_norm
            self.norm = PreBatchNormTime(config.channel_dim)
        elif config.norm_placement == NormPlacement.POST:
            self._forward = self._forward_post_norm
            self.norm = nn.LayerNorm([config.time_dim, config.channel_dim])
        else:
            self._forward = self._forward_no_norm

    def _core(self, x):
        '''
        Fonction centrale de mélange temporel.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        y = x.permute(0, 2, 1)  # [Batch, Channel, Time_in (T)]
        # --- MLP time (from TSMixer paper) ---
        y = self.fc(y)       # [Batch, Channel, Time_out (T)]
        y = self.activation(y)
        #dropout values
        y = self.dropout(y)
        # -------------------------------------
        y = y.permute(0, 2, 1)  # [Batch, Time_out (T), Channel]
        return y
    
    def _forward_pre_norm(self, x):
        '''
        Mélange temporel avec normalisation pré-activation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        y = self._core(self.norm(x))
        return x+y

    def _forward_post_norm(self, x):
        '''
        Mélange temporel avec normalisation post-activation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        y = self._core(x)
        return self.norm(x+y)

    def _forward_no_norm(self, x):
        '''
        Mélange temporel sans normalisation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        y = self._core(x)
        return x+y

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        return self._forward(x)
    
@register_model('TSM_FeatureMixing')
class TSM_FeatureMixing(BaseModel):
    '''
    Module de mélange de caractéristiques pour combiner les informations à travers les différentes étapes du TSMixer.
    [Batch, Time (T), Channel] -> [Batch, Time (T), Channel]
    '''
    def __init__(self, config: TSM_MixingConfig):
        super(TSM_FeatureMixing, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(config.channel_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)
        self.activation = config.activation
        self.dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate is not None else nn.Identity()

        # Projection OPTIONNELLE si output_dim != channel_dim
        # Permet de faire du "feature mixing" avec changement de dimension 'feature'
        if config.output_dim != config.channel_dim:
            self.projection = nn.Linear(config.channel_dim, config.output_dim)
        else:
            self.projection = nn.Identity()

        # --- Selectionner la function 'forward' en fonction du placement de la normalisation ---
        if config.norm_placement == NormPlacement.PRE:
            self._forward = self._forward_pre_norm
            self.norm = PreBatchNormTime(config.channel_dim)
        elif config.norm_placement == NormPlacement.POST:
            self._forward = self._forward_post_norm
            self.norm = nn.LayerNorm([config.time_dim, config.output_dim])
        else:
            self._forward = self._forward_no_norm

    def _core(self, x):
        '''
        Fonction centrale de mélange de caractéristiques.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), Channel]
        '''
        # --- MLP feat (from TSMixer paper) ---
        y = self.fc1(x)          # [Batch, Time (T), Hidden_dim]
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)          # [Batch, Time (T), Channel]
        y = self.dropout(y)
        # -------------------------------------
        return y
    
    def _forward_pre_norm(self, x):
        '''
        Mélange de caractéristiques avec normalisation pré-activation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), output_dim]
        '''
        y = self._core(self.norm(x))
        return self.projection(x)+y
    
    def _forward_post_norm(self, x):
        '''
        Mélange de caractéristiques avec normalisation post-activation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), output_dim]
        '''
        y = self._core(x)
        y = self.projection(x)+y
        return self.norm(y)
    
    def _forward_no_norm(self, x):
        '''
        Mélange de caractéristiques sans normalisation.
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), output_dim]
        '''
        y = self._core(x)
        return self.projection(x)+y


    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), output_dim]
        '''
        return self._forward(x)

@register_model('TSM_MixingLayer')
class TSM_MixingLayer(BaseModel):
    '''
    Couche de mélange pour combiner les modules de mélange temporel et de mélange de caractéristiques.
    [Batch, Time (T), Channel] -> (TimeMixing -> FeatureMixing) -> [Batch, Time (T), output_dim]
    '''
    def __init__(self, config: TSM_MixingConfig):
        super(TSM_MixingLayer, self).__init__()
        self.config = config

        self.time_mixing = TSM_TimeMixing(config)
        self.feature_mixing = TSM_FeatureMixing(config)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [Batch, Time (T), Channel]
        Returns:
            torch.Tensor: [Batch, Time (T), output_dim]
        '''
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x

class Detailled_hidden_dim_Config:
    def __init__(self, historical_hidden_dim: int, future_hidden_dim: int, mixing_hidden_dim: int):
        self.historical_hidden_dim = historical_hidden_dim
        self.future_hidden_dim = future_hidden_dim
        self.mixing_hidden_dim = mixing_hidden_dim

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Detailled_hidden_dim_Config':
        return cls(
            historical_hidden_dim=config_dict['historical_hidden_dim'],
            future_hidden_dim=config_dict['future_hidden_dim'],
            mixing_hidden_dim=config_dict['mixing_hidden_dim']
        )
    
    def to_dict(self) -> dict:
        return {
            'historical_hidden_dim': self.historical_hidden_dim,
            'future_hidden_dim': self.future_hidden_dim,
            'mixing_hidden_dim': self.mixing_hidden_dim
        }

class TSMixerExt_Config:
    '''
    Configuration pour le TSMixerExt.
    Args:
        historical_time_dim (int): Dimension temporelle des données historiques d'entrée.\n
            Cela correspond à "lookback window"
        historical_channel_dim (int): Dimension des canaux des données historiques d'entrée.
        future_time_dim (int): Dimension temporelle des données futures d'entrée.\n
            Cela correspond à "horizon" de prévision.
        future_channel_dim (int): Dimension des canaux des données futures d'entrée.
        mixing_hidden_dim (int | Detailled_hidden_dim_Config): Dimension de la couche cachée dans les modules de mélange du TSMixer.
        k_mixing_layers (int): Nombre de couches de mélange temporel et de mélange de caractéristiques à empiler dans le TSMixer.\n
            minimum 1 (une couche de mélange temporel + mélange de caractéristiques)
        output_channel_dim (int): Dimension des canaux de sortie du TSMixerExt.\n
            Cela correspond à la dimension 'feature' de sortie du TSMixerExt\n
            Si None, output_channel_dim = future_channel_dim
        norm_placement (NormPlacement): Placement de la normalisation dans les modules de mélange du TSMixer.\n
            PRE: 2d batch normalisation avant l'activation (comme dans le TSMixer original)\n
            POST: 2d layer normalisation après l'activation\n
            NONE: pas de normalisation
        activation (nn.Module): Fonction d'activation dans les modules de mélange du TSMixer.
        dropout_rate (float | None): Taux de dropout dans les modules de mélange du TSMixer.
    '''
    def __init__(
            self,
            historical_time_dim: int,
            historical_channel_dim: int,
            future_time_dim: int,
            future_channel_dim: int,
            mixing_hidden_dim: int | Detailled_hidden_dim_Config = 64,
            k_mixing_layers: int = 1,
            output_channel_dim: int | None = None,
            norm_placement: NormPlacement = NormPlacement.PRE,
            activation: nn.Module = nn.ReLU(),
            dropout_rate: float | None = None,
        ):
        '''
        Args:
            historical_time_dim (int): Dimension temporelle des données historiques d'entrée.\n
                Cela correspond à "lookback window"
            historical_channel_dim (int): Dimension des canaux des données historiques d'entrée.
            future_time_dim (int): Dimension temporelle des données futures d'entrée.\n
                Cela correspond à "horizon" de prévision.
            future_channel_dim (int): Dimension des canaux des données futures d'entrée.
            mixing_hidden_dim (int | Detailled_hidden_dim_Config): Dimension de la couche cachée dans les modules de mélange du TSMixer.
            k_mixing_layers (int): Nombre de couches de mélange temporel et de mélange de caractéristiques à empiler dans le TSMixer.\n
                minimum 1 (une couche de mélange temporel + mélange de caractéristiques)
            output_channel_dim (int | None): Dimension des canaux de sortie du TSMixerExt.\n
                Cela correspond à la dimension 'feature' de sortie du TSMixerExt\n
                Si None, output_channel_dim = future_channel_dim
            norm_placement (NormPlacement): Placement de la normalisation dans les modules de mélange du TSMixer.\n
                PRE: 2d batch normalisation avant l'activation (comme dans le TSMixer original)\n
                POST: 2d layer normalisation après l'activation\n
                NONE: pas de normalisation
            activation (nn.Module): Fonction d'activation dans les modules de mélange du TSMixer.
            dropout_rate (float | None): Taux de dropout dans les modules de mélange du TSMixer.
        '''
        self.historical_time_dim = historical_time_dim
        self.historical_channel_dim = historical_channel_dim
        self.future_time_dim = future_time_dim
        self.future_channel_dim = future_channel_dim
        self.mixing_hidden_dim = mixing_hidden_dim
        self.k_mixing_layers = k_mixing_layers if k_mixing_layers >= 1 else 1
        self.output_channel_dim = output_channel_dim if output_channel_dim is not None else future_channel_dim
        self.norm_placement = norm_placement
        self.activation = activation
        self.dropout_rate = dropout_rate

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TSMixerExt_Config':
        '''
        Crée une instance de TSMixerExt_Config à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            TSMixerExt_Config: Instance configurée.
        '''
        if isinstance(config_dict['mixing_hidden_dim'], dict):
            mixing_hidden_dim = Detailled_hidden_dim_Config.from_dict(config_dict['mixing_hidden_dim'])
        else:
            mixing_hidden_dim = config_dict.get('mixing_hidden_dim', 64)

        norm_placement = NormPlacement(config_dict['norm_placement'])
        
        activation = None
        act_name = config_dict.get('activation', None)
        act_kwargs = config_dict.get('activation_kwargs', {})
        if act_name is not None:
            assert act_name in act_f.ACTIVATION_REGISTRY, f"Activation '{act_name}' non reconnue dans le registre de function d'activation."
            activation = act_f.ACTIVATION_REGISTRY[act_name](**act_kwargs)

        return cls(
            historical_time_dim=config_dict['historical_time_dim'],
            historical_channel_dim=config_dict['historical_channel_dim'],
            future_time_dim=config_dict['future_time_dim'],
            future_channel_dim=config_dict['future_channel_dim'],
            mixing_hidden_dim=mixing_hidden_dim,
            k_mixing_layers=config_dict.get('k_mixing_layers', 1),
            output_channel_dim=config_dict.get('output_channel_dim', None),
            norm_placement=norm_placement,
            activation=activation,
            dropout_rate=config_dict.get('dropout_rate', None)
        )
    
    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du TSMixerExt en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return {
            'historical_time_dim': self.historical_time_dim,
            'historical_channel_dim': self.historical_channel_dim,
            'future_time_dim': self.future_time_dim,
            'future_channel_dim': self.future_channel_dim,
            'mixing_hidden_dim': (
                    self.mixing_hidden_dim.to_dict()
                    if isinstance(self.mixing_hidden_dim, Detailled_hidden_dim_Config)
                    else self.mixing_hidden_dim
                ),
            'k_mixing_layers': self.k_mixing_layers,
            'output_channel_dim': self.output_channel_dim,
            'norm_placement': self.norm_placement.value,
            'activation': self.activation.__class__.__name__,
            'activation_kwargs': (
                    self.activation.__kwargs__
                    if self.activation is not None and hasattr(self.activation, '__kwargs__')
                    else {}
                ),
            'dropout_rate': self.dropout_rate
        }

@register_model('TSMixerExt')
class TSMixerExt(BaseModel):
    '''
    TSMixerExt est la version de TSMixer qui prend en charge les données historiques et futures en entrée.
    À la diférence de TSMixerEXT du papier original, celui ci ne prend pas d'entrées "statiques".
    '''
    def __init__(self, config: TSMixerExt_Config):
        super(TSMixerExt, self).__init__()
        self.config = config

        if isinstance(config.mixing_hidden_dim, Detailled_hidden_dim_Config):
            historical_hidden_dim = config.mixing_hidden_dim.historical_hidden_dim
            future_hidden_dim = config.mixing_hidden_dim.future_hidden_dim
            mixing_hidden_dim = config.mixing_hidden_dim.mixing_hidden_dim
        else:
            historical_hidden_dim = config.mixing_hidden_dim
            future_hidden_dim = config.mixing_hidden_dim
            mixing_hidden_dim = config.mixing_hidden_dim

        # *** ALIGN STAGE *** --------------------------------------------------------------------
        # *** Historical ***
        self.historical_projection = TSM_TemporalProjection(config.historical_time_dim, config.future_time_dim)
        self.historical_feature_mixing = TSM_FeatureMixing(
            TSM_MixingConfig(
                time_dim = config.future_time_dim,
                channel_dim = config.historical_channel_dim,
                hidden_dim = historical_hidden_dim, # Latent space dimension for historical feature mixing
                output_dim = historical_hidden_dim, # La sortie est de dimension [batch, future_time_dim, latent_dim]
                norm_placement = config.norm_placement,
                activation = config.activation,
                dropout_rate = config.dropout_rate 

        ))
        # *** Future ***
        self.future_feature_mixing = TSM_FeatureMixing(
            TSM_MixingConfig(
                time_dim = config.future_time_dim,
                channel_dim = config.future_channel_dim,
                hidden_dim = future_hidden_dim, # Latent space dimension for future feature mixing
                output_dim = future_hidden_dim, # La sortie est de dimension [batch, future_time_dim, latent_dim]
                norm_placement = config.norm_placement,
                activation = config.activation,
                dropout_rate = config.dropout_rate
        ))
        # ----------------------------------------------------------------------------------------
        # *** MIXING STAGE *** -------------------------------------------------------------------
        self.mixing_layers = nn.ModuleList([
            TSM_MixingLayer(TSM_MixingConfig(
                time_dim = self.future_feature_mixing.config.time_dim,
                channel_dim = (
                    # La concaténation des sorties historique et future donne une dimension de 2*latent_dim
                    self.historical_feature_mixing.config.output_dim + self.future_feature_mixing.config.output_dim if k==0
                    else
                    # Les couches de mélange suivantes prennent en entrée la dimension de sortie de la couche précédente
                    mixing_hidden_dim
                    ),
                hidden_dim = mixing_hidden_dim,
                output_dim = mixing_hidden_dim, # La sortie est de dimension [batch, future_time_dim, latent_dim]
                norm_placement = config.norm_placement,
                activation = config.activation,
                dropout_rate = config.dropout_rate
            )) for k in range(config.k_mixing_layers)
        ])
        # ----------------------------------------------------------------------------------------
        # *** OUTPUT PROJECTION *** --------------------------------------------------------------
        self.output_projection = nn.Linear(self.mixing_layers[-1].config.output_dim, config.output_channel_dim)
        # ----------------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TSMixerExt':
        '''
        Crée une instance de TSMixerExt à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            TSMixerExt: Instance configurée.
        '''
        mixer_config = TSMixerExt_Config.from_dict(config_dict)
        return cls(mixer_config)
    
    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du TSMixerExt en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return self.config.to_dict()

    def forward(self, historical_x, future_x):
        '''
        Args:
            historical_x (torch.Tensor): [Batch, Historical_Time (L), Historical_Channel]
            future_x (torch.Tensor): [Batch, Future_Time (T), Future_Channel]
        Returns:
            torch.Tensor: [Batch, Future_Time (T), Mixing_Output_Dim]
        '''
        # *** ALIGN STAGE ***
        historical_x = self.historical_projection(historical_x) # [Batch, Future_Time (T), Historical_Channel]
        historical_x = self.historical_feature_mixing(historical_x) # [Batch, Future_Time (T), Latent_Dim]

        future_x = self.future_feature_mixing(future_x) # [Batch, Future_Time (T), Latent_Dim]

        # Concaténer les informations historiques et futures alignées
        x = torch.cat([historical_x, future_x], dim=-1) # [Batch, Future_Time (T), 2*Latent_Dim]

        # *** MIXING STAGE ***
        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x) # [Batch, Future_Time (T), Latent_Dim]

        x = self.output_projection(x) # [Batch, Future_Time (T), Output_Channel_Dim]

        return x
    
@register_model('TSMixerExt_PositiveOutput')
class TSMixerExt_PositiveOutput(TSMixerExt):
    '''
    Variante de TSMixerExt avec une projection de sortie contraignant les prédictions à être positives.
    '''
    def __init__(self, config: TSMixerExt_Config):
        super(TSMixerExt_PositiveOutput, self).__init__(config)
        # Remplacer la projection de sortie par une projection avec activation ReLU pour garantir des sorties positives
        self.output_projection = nn.Sequential(
            nn.Linear(self.mixing_layers[-1].config.output_dim, config.output_channel_dim),
            act_f.PositiveClip()
        )

@register_model('TSMixerExt_2TimeIngress')
class TSMixerExt_2TimeIngress(BaseModel):
    '''
    Variante de TSMixerExt avec une architecture d'entrée de données historiques en 2 tensors séparés : un pour les 'features' (en heure) et un pour des 'features' (en jour).
    Alors cela fait 3 tensors d'entrée : historical_hourly_x, historical_daily_x, future_hourly_x.
    L'idée est de permettre un traitement différencié des données en heure et des données en jour, qui ont des caractéristiques temporelles différentes (heures vs jours).
    La partie historique subit:
    - Features historiques en heure (historical_hourly_x) suibit transformation -> time mixing + feature mixing --(aka TSM_MixingLayer)--> latent space historique (en heure)
    - Latent space historique (en heure) -> TSM_TemporalProjection (permet condenser les heurs en jours) -> latent space historique (en jour)
    - Puis le tensor [latent space historique (en jour)] est concatené avec historical_daily_x (features historiques en jour) pour former un tensor historique enrichi
    La partie future subit:
    - Features futures en heure (future_hourly_x) suibit transformation -> time mixing + feature mixing --(aka TSM_MixingLayer)--> latent space futur (en heure)
    - Latent space futur (en heure) -> TSM_TemporalProjection (permet condenser les heurs en jours) -> latent space futur (en jour)
    - !!! Aucune concaténation avec des features futures en jour, car on suppose que les données futures sont uniquement disponibles en heure (ex: prévisions météo horaires) !!! La partie journalière est ce qu'on cherche à prédire

    Nous obtenons donc 2 tensors:
    - historical_hourly_x (transformé en latent space en jour) concatené avec historical_daily_x ---> Nouveau tensor historique enrichi (en jour)
    - future_hourly_x (transformé en latent space en jour) ---> Nouveau tensor futur enrichi (en jour)
    
    Puis ces tensors sont passé au modèle TSMixerExt_PositiveOutput pour le mélange temporel et de caractéristiques, et la projection de sortie.

    Args:
        config_HistoryIngress (TSM_MixingConfig): Configuration pour la partie d'ingress temporel "historique" du TSMixerExt_2TimeIngress, qui traite les données historiques en heure et les projette dans un espace latent en jour.
        config_FutureIngress (TSM_MixingConfig): Configuration pour la partie d'ingress temporel "futur" du TSMixerExt_2TimeIngress, qui traite les données futures en heure et les projette dans un espace latent en jour.
        config_TSMixerExt (TSMixerExt_Config): Configuration pour le sous modèle TSMixerExt_PositiveOutput qui réalise le mélange temporel et de caractéristiques, et la projection de sortie du TSMixerExt_2TimeIngress.
    '''
    def __init__(self, config_HistoryIngress: TSM_MixingConfig, config_FutureIngress: TSM_MixingConfig, config_TSMixerExt: TSMixerExt_Config):
        super(TSMixerExt_2TimeIngress, self).__init__()
        self.config_HistoryIngress = config_HistoryIngress
        self.config_FutureIngress = config_FutureIngress
        self.config_TSMixerExt = config_TSMixerExt

        # *** History Ingress ***
        self.history_hourly_mixing = TSM_MixingLayer(config_HistoryIngress)
        self.history_hourly_projection = TSM_TemporalProjection(config_HistoryIngress.time_dim, config_TSMixerExt.historical_time_dim)

        # *** Future Ingress ***
        self.future_hourly_mixing = TSM_MixingLayer(config_FutureIngress)
        self.future_hourly_projection = TSM_TemporalProjection(config_FutureIngress.time_dim, config_TSMixerExt.future_time_dim)

        # *** TSMixerExt_PositiveOutput ***
        self.tsmixer_ext = TSMixerExt_PositiveOutput(config_TSMixerExt)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TSMixerExt_2TimeIngress':
        '''
        Crée une instance de TSMixerExt_2TimeIngress à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            TSMixerExt_2TimeIngress: Instance configurée.
        '''
        config_HistoryIngress = TSM_MixingConfig.from_dict(config_dict.get("config_HistoryIngress", {}))
        config_FutureIngress = TSM_MixingConfig.from_dict(config_dict.get("config_FutureIngress", {}))
        config_TSMixerExt = TSMixerExt_Config.from_dict(config_dict.get("config_TSMixerExt", {}))
        return cls(config_HistoryIngress, config_FutureIngress, config_TSMixerExt)

    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du TSMixerExt_2TimeIngress en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return {
            "config_HistoryIngress": self.config_HistoryIngress.to_dict(),
            "config_FutureIngress": self.config_FutureIngress.to_dict(),
            "config_TSMixerExt": self.config_TSMixerExt.to_dict()
        }
    
    def forward(self, historical_hourly_x, historical_daily_x, future_hourly_x):
        '''
        Args:
            historical_hourly_x (torch.Tensor): [Batch, Historical_Hourly_Time (Lh), Historical_Hourly_Channel]
            historical_daily_x (torch.Tensor): [Batch, Historical_Daily_Time (Ld), Historical_Daily_Channel]
            future_hourly_x (torch.Tensor): [Batch, Future_Hourly_Time (Lf), Future_Hourly_Channel]
        Returns:
            torch.Tensor: [Batch, Future_Daily_Time (Td), Output_Channel_Dim]
        '''
        # *** History Ingress ***
        historical_x = self.history_hourly_mixing(historical_hourly_x) # [Batch, Historical_Hourly_Time (Lh), Latent_Dim]
        historical_x = self.history_hourly_projection(historical_x) # [Batch, Future_Time (T), Latent_Dim]

        # Concaténer les features historiques en jour avec le latent space projeté des features historiques en heure
        historical_x = torch.cat([historical_x, historical_daily_x], dim=-1) # [Batch, Future_Time (T), Latent_Dim + Historical_Daily_Channel]

        # *** Future Ingress ***
        future_x = self.future_hourly_mixing(future_hourly_x) # [Batch, Future_Hourly_Time (Lf), Latent_Dim]
        future_x = self.future_hourly_projection(future_x) # [Batch, Future_Time (T), Latent_Dim]

        # *** TSMixerExt_PositiveOutput ***
        x = self.tsmixer_ext(historical_x, future_x) # [Batch, Future_Time (T), Output_Channel_Dim]

        return x