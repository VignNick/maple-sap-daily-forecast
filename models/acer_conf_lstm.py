import torch
import torch.nn as nn
import models.activations as act_f
from .model import register_model, BaseModel
from .mlp_block import MLPBlock

@register_model('AcerConfLSTM')
class AcerConfLSTM(BaseModel):
    def __init__(
            self,
            lstm_input_size, lstm_hidden_size = 512, lstm_num_layers = 2, lstm_dropout=0.2,
            lstm_lookback=3, horizon=1, temperature_forecast=True,
            mlp_config: list[MLPBlock.SequentialBlock] | None = None
        ):
        super(AcerConfLSTM, self).__init__()

        if lstm_lookback < 1:
            lstm_lookback = 1  # Assurez-vous que lstm_lookback est au moins 1

        # !!!!! None est valide car un mlp_block sans configuration génère un MLP avec une seule couche de input_dim -> output_dim, ce qui est un comportement raisonnable par défaut. !!!!!! 
        # if mlp_config is None:
        #     mlp_config = [MLPBlock.SequentialBlock(128, layer_norm=True, activation=nn.SiLU(), dropout_rate=0.2)]

        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_lookback = lstm_lookback
        self.horizon = horizon
        self.temperature_forecast = temperature_forecast

        # LSTM pour traiter les séquences d'entrée
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        # LayerNorm pour l'état caché du LSTM
        self.layer_norm = nn.LayerNorm(lstm_hidden_size * lstm_lookback)

        # MLP pour traiter les données de forecast
        self.mlp = MLPBlock(
            input_dim=lstm_hidden_size * lstm_lookback + 24*horizon*(int(temperature_forecast)),
            output_dim=horizon,
            mlp_config=mlp_config
        )

        # --- Contrainte positive sur la sortie ---
        self.positive_clip = act_f.PositiveClip()

    def forward(self, x_seq, temp_forecast):
        """
        Args:
            x_seq (torch.Tensor): (batch_size, seq_len, input_size)
            temp_forecast (torch.Tensor): (batch_size, 24*horizon) forecast -> T_h Forecast\n
                Si Forecast désactivé, fournir un tenseur de taille (batch_size, 0)
        """
        # Obtenir toutes les sorties cachées du LSTM
        output, (_, _) = self.lstm(x_seq) # output: (batch_size, seq_len, hidden_size)

        # Obtenir les "lookback" dernières sorties cachées
        selected_outputs = output[:, -self.lstm_lookback:, :]  # (batch_size, lookback, hidden_size)

        # Redimensionner pour la concaténation
        selected_outputs = selected_outputs.contiguous().view(x_seq.size(0), -1)  # (batch_size, lookback * hidden_size)

        # Appliquer la normalisation de couche
        selected_outputs = self.layer_norm(selected_outputs)

        # Concatenate forecast temperature
        combined = torch.cat([selected_outputs, temp_forecast], dim=1)

        raw_out  = self.mlp(combined)
        out = self.positive_clip(raw_out)
        return out
    
    def to_dict(self) -> dict:
        '''
        Sérialise la configuration du modèle en un dictionnaire.
        Returns:
            dict: Dictionnaire contenant les paramètres de configuration.
        '''
        return {
            'lstm_input_size': self.lstm_input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_num_layers': self.lstm_num_layers,
            'lstm_dropout': self.lstm_dropout,
            'lstm_lookback': self.lstm_lookback,
            'horizon': self.horizon,
            'temperature_forecast': self.temperature_forecast,
            'mlp_config': self.mlp.to_dict() if self.mlp is not None else None
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AcerConfLSTM':
        '''
        Crée une instance de AcerConfLSTM à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres de configuration.
        Returns:
            AcerConfLSTM: Instance configurée.
        '''
        mlp_config = MLPBlock.from_dict_get_mlp_config(config_dict) if config_dict.get('mlp_config') is not None else None
        return cls(
            lstm_input_size=config_dict['lstm_input_size'],
            lstm_hidden_size=config_dict['lstm_hidden_size'],
            lstm_num_layers=config_dict['lstm_num_layers'],
            lstm_dropout=config_dict['lstm_dropout'],
            lstm_lookback=config_dict['lstm_lookback'],
            horizon=config_dict['horizon'],
            temperature_forecast=config_dict['temperature_forecast'],
            mlp_config=mlp_config
        )