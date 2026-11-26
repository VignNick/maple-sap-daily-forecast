import torch
from .model import register_model, BaseModel
from .MeteoProcess import MeteoCNNAttentionEmbedding
from .TSMixer import TSMixerExt_PositiveOutput as TSMixerExt_PO, TSMixerExt_Config
from utils.yaml.YAML_registry import yaml_serializable
from utils.yaml.YAML_factory import make_yaml_safe, restore_yaml_safe

@yaml_serializable()
class MeteoEmbConfig:
    def __init__(
        self,
        feature_dim: int,
        conv_dim: int = 64,
        num_conv_layers: int = 3,
        attention_dim: int = 64,
        num_attention_heads: int = 4,
        output_dim: int = 128
    ):
        self.feature_dim = feature_dim
        self.conv_dim = conv_dim
        self.num_conv_layers = num_conv_layers
        self.attention_dim = attention_dim
        self.num_attention_heads = num_attention_heads
        self.output_dim = output_dim
        

@register_model('MeteoTSMixer')
class MeteoTSMixer(BaseModel):
    def __init__(
        self,
        MeteoEmbConfig: MeteoEmbConfig,
        TSMixerConfig: TSMixerExt_Config,
    ):
        super(MeteoTSMixer, self).__init__()

        self.MeteoEmbConfig = MeteoEmbConfig
        self.TSMixerConfig = TSMixerConfig

        self.meteo_emb = MeteoCNNAttentionEmbedding(
            feature_dim=MeteoEmbConfig.feature_dim,
            conv_dim=MeteoEmbConfig.conv_dim,
            num_conv_layers=MeteoEmbConfig.num_conv_layers,
            attention_dim=MeteoEmbConfig.attention_dim,
            num_attention_heads=MeteoEmbConfig.num_attention_heads,
            output_dim=MeteoEmbConfig.output_dim
        )

        # self.meteo_emb_test = self.meteo_emb

        self.tsmixer = TSMixerExt_PO(
            config=TSMixerConfig
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MeteoTSMixer':
        try:
            tsmixer_config = restore_yaml_safe(config_dict['TSMixerConfig'])
        except Exception as e:
            tsmixer_config = TSMixerExt_Config.from_dict(config_dict['TSMixerConfig']['value'])

        return cls(
            MeteoEmbConfig=restore_yaml_safe(config_dict['MeteoEmbConfig']),
            TSMixerConfig=tsmixer_config
        )

    def to_dict(self) -> dict:
        return {
            "MeteoEmbConfig": make_yaml_safe(self.MeteoEmbConfig),
            "TSMixerConfig": make_yaml_safe(self.TSMixerConfig)
        }
    
    def forward(self, historical_meteo_hourly_x, historical_sap_daily_x, future_meteo_hourly_x):
        '''
        Args:
            historical_meteo_hourly_x (torch.Tensor): [Batch, Journee, Heure, Feature]
            historical_sap_daily_x (torch.Tensor): [Batch, Journee, Feature]
            future_meteo_hourly_x (torch.Tensor): [Batch, Journee, Heure, Feature]
        Returns:
            torch.Tensor: [Batch, Journee, output_dim]
        '''

        # ---- 1. Embedding météo ----
        historical_meteo_emb = self.meteo_emb(historical_meteo_hourly_x)  # [B, J, emb_dim]
        future_meteo_emb = self.meteo_emb(future_meteo_hourly_x)  # [B, J, emb_dim]

        # ---- 2. Concat SAP quotidien aux embeddings météo historiques ----
        historical_meteo_emb = torch.cat([historical_meteo_emb, historical_sap_daily_x], dim=-1)  # [B, J, emb_dim + sap_feature]

        # ---- 3. TSMixerExt ----
        output = self.tsmixer(
            historical_meteo_emb,
            future_meteo_emb
        )  # [B, J, output_dim]

        return output

@register_model('MeteoTSMixer_withDailyVar')
class MeteoTSMixer_withDailyVar(MeteoTSMixer):
    def __init__(
        self,
        MeteoEmbConfig: MeteoEmbConfig,
        TSMixerConfig: TSMixerExt_Config,
    ):
        super(MeteoTSMixer_withDailyVar, self).__init__(
            MeteoEmbConfig=MeteoEmbConfig,
            TSMixerConfig=TSMixerConfig
        )
        del self.meteo_emb

        self.meteo_emb_hist = MeteoCNNAttentionEmbedding(
                    feature_dim=MeteoEmbConfig.feature_dim,
                    conv_dim=MeteoEmbConfig.conv_dim,
                    num_conv_layers=MeteoEmbConfig.num_conv_layers,
                    attention_dim=MeteoEmbConfig.attention_dim,
                    num_attention_heads=MeteoEmbConfig.num_attention_heads,
                    output_dim=MeteoEmbConfig.output_dim
                )

        self.meteo_emb_future = MeteoCNNAttentionEmbedding(
            feature_dim=MeteoEmbConfig.feature_dim,
            conv_dim=MeteoEmbConfig.conv_dim,
            num_conv_layers=MeteoEmbConfig.num_conv_layers,
            attention_dim=MeteoEmbConfig.attention_dim,
            num_attention_heads=MeteoEmbConfig.num_attention_heads,
            output_dim=MeteoEmbConfig.output_dim
        )


    def forward(self, historical_meteo_hourly_x, historical_sap_daily_x, future_meteo_hourly_x, future_daily_var_x):
        '''
        Args:
            historical_meteo_hourly_x (torch.Tensor): [Batch, Journee, Heure, Feature]
            historical_sap_daily_x (torch.Tensor): [Batch, Journee, Feature]
            future_meteo_hourly_x (torch.Tensor): [Batch, Journee, Heure, Feature]
            future_daily_var_x (torch.Tensor): [Batch, Journee, Feature]
        Returns:
            torch.Tensor: [Batch, Journee, output_dim]
        '''

        # ---- 1. Embedding météo ----
        historical_meteo_emb = self.meteo_emb_hist(historical_meteo_hourly_x)  # [B, J, emb_dim]
        future_meteo_emb = self.meteo_emb_future(future_meteo_hourly_x)  # [B, J, emb_dim]

        # ---- 2. Concat SAP quotidien aux embeddings météo historiques ----
        historical_meteo_emb = torch.cat([historical_meteo_emb, historical_sap_daily_x], dim=-1)  # [B, J, emb_dim + sap_feature]
        future_meteo_emb = torch.cat([future_meteo_emb, future_daily_var_x], dim=-1)  # [B, J, emb_dim + daily_var_feature]

        # ---- 3. TSMixerExt ----
        output = self.tsmixer(
            historical_meteo_emb,
            future_meteo_emb,
        )  # [B, J, output_dim]

        return output