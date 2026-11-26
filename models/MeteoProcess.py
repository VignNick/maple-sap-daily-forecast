import torch.nn as nn
from .model import register_model, BaseModel

@register_model('MeteoCNNAttentionEmbedding')
class MeteoCNNAttentionEmbedding(BaseModel):
    def __init__(
        self,
        feature_dim: int,
        conv_dim: int = 64,
        num_conv_layers: int = 3,
        attention_dim: int = 64,
        num_attention_heads: int = 4,
        output_dim: int = 128
    ):
        super(MeteoCNNAttentionEmbedding, self).__init__()

        self.feature_dim = feature_dim
        self.conv_dim = conv_dim
        self.num_conv_layers = num_conv_layers
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.output_dim = output_dim

        # ----------------------------------------------------------------------------------------
        # *** 1. CNN temporel pour extraire les caractéristiques locales *** ---------------------
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, conv_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            *[nn.Sequential(
                nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
                nn.SiLU()
            ) for _ in range(num_conv_layers - 1)]
        )

        # ----------------------------------------------------------------------------------------
        # *** 2. Projection linéaire pour préparer les données pour l'attention *** --------------
        self.projection = nn.Linear(conv_dim, attention_dim)

        # ----------------------------------------------------------------------------------------
        # *** 3. Attention temporelle (intra-journée) *** ----------------------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )

        # ----------------------------------------------------------------------------------------
        # *** 4. Pooling et projection finale pour obtenir l'embedding de sortie *** -------------
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(attention_dim, output_dim)

        # ----------------------------------------------------------------------------------------
        # *** 5. Normalisation *** ---------------------------------------------------------------
        self.norm = nn.LayerNorm(output_dim)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MeteoCNNAttentionEmbedding':
        return cls(
            feature_dim=config_dict['feature_dim'],
            conv_dim=config_dict.get('conv_dim', 64),
            num_conv_layers=config_dict.get('num_conv_layers', 3),
            attention_dim=config_dict.get('attention_dim', 64),
            num_attention_heads=config_dict.get('num_attention_heads', 4),
            output_dim=config_dict.get('output_dim', 128)
        )
        
    def to_dict(self) -> dict:
        return {
            'feature_dim': self.feature_dim,
            'conv_dim': self.conv_dim,
            'num_conv_layers': self.num_conv_layers,
            'attention_dim': self.attention_dim,
            'num_attention_heads': self.num_attention_heads,
            'output_dim': self.output_dim
        }
    
    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): [Batch, Journee, Heure, Feature]
        Returns:
            torch.Tensor: [Batch, Journee, output_dim]
        '''

        B, J, H, F = x.shape

        # ---- 1. CNN temporel ----
        x = x.view(-1, H, F)  # [B*J, H, F]
        x = x.permute(0, 2, 1)  # [B*J, F, H]

        x = self.conv(x)  # [B*J, conv_dim, H]

        x = x.permute(0, 2, 1)  # [B*J, H, conv_dim]

        # ---- 2. Projection linéaire ----
        x = self.projection(x)  # [B*J, H, attention_dim]

        # ---- 3. Attention temporelle ----
        x, _ = self.attention(x, x, x)  # [B*J, H, attention_dim]

        # ---- 4. Pooling et projection finale ----
        x = x.permute(0, 2, 1)  # [B*J, attention_dim, H]
        x = self.pooling(x)  # [B*J, attention_dim, 1]
        x = x.squeeze(-1)  # [B*J, attention_dim]

        x = self.output_projection(x)  # [B*J, output_dim]

        #--- 5. Normalisation ---
        x = self.norm(x)  # [B*J, output_dim]

        return x.view(B, J, -1)  # [B, J, output_dim]