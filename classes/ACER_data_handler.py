import torch
import re
from .maple_helper import NormHelper
from . import ACER_helper
from enum import Enum

'''
ACER_data_handler.py, par Nicolas Vigneault janvier 2026
Ce module contient la classe ACERDataModule, qui gère les données ACER pour l'entraînement et la validation des modèles.
Il fournit des méthodes pour générer des séquences de données, diviser les données en ensembles d'entraînement et de validation,
 normaliser les données et récupérer des tenseurs spécifiques pour l'entraînement des modèles.

La classe ACERDataModule est conçue pour être flexible et configurable, permettant de définir
 divers paramètres de génération de données, de normalisation et de retour des champs.
Elle est également capable de sérialiser et de désérialiser ses paramètres à partir d'un dictionnaire,
facilitant ainsi la sauvegarde et le chargement des configurations.

La classe ACERDataset(Dataset) est un wrapper autour du DataModule pour fournir un accès aux données sous forme de tenseurs PyTorch.
Elle permet de charger les données en mémoire (CPU et GPU) et de les manipuler facilement pour l'entraînement des modèles.
'''

class ACERDataModule:
    """
    Gestionnaire de données pour ACER.
    Propriétaire unique des données CPU + stats de normalisation.
    """

    def __init__(
        self,
        ACER_csv_path: str = "./Data/ACER_2000-2024.csv",
        n_lookback_days: int = 3,
        horizon: int = 1,
        start_padding_days: int = 2,
        end_padding_days: int = 0,
        norm_by_dayly_deviation: bool = False,
        sample_hour: int = 23,
        augment_data: bool = False,
        year_no_augment: list[int] = [],
        val_split: float | list[int] = 0.2,
        normalize_volume: bool = False,
        normalize_temperature: bool = False,
        scaling_mode_robust: bool = False,
        scaling_per_feature: bool = False,
        return_fields: list[str] | None = None,
        take_only_first_system: bool = False
    ):
        # Code de cache pour suivre les changements aux données/paramètres CPU
        self._cached_code = 0

        self.ACER_df, self.past_years_df = ACER_helper.get_ACERdf_PastYearsDf(ACER_csv_path)

        # Data Parameters
        self.n_lookback_days = n_lookback_days
        self.horizon = horizon
        self.start_padding_days = start_padding_days
        self.end_padding_days = end_padding_days
        self.norm_by_dayly_deviation = norm_by_dayly_deviation
        self.sample_hour = sample_hour
        self.augment_data = augment_data
        self.year_no_augment = year_no_augment
        self.take_only_first_system = take_only_first_system
        # Data Storage
        self.allData_df = None
        self._allData_upToDate = False
        self.length = 0

        # Train/Val Split
        self._val_split = val_split
        self._train_idx = None
        self._val_idx = None
        self._split_upToDate = False

        # Normalization flags and stats
        self.normalize_volume = normalize_volume
        self.normalize_temperature = normalize_temperature
        self.scaling_mode_robust = scaling_mode_robust
        self.scaling_per_feature = scaling_per_feature
        self.normalization_stats = {}
        self._fitted = False

        # Return fields for __getitem__
        self.return_fields = []
        self.set_return_fields(return_fields)

        # Relation avec les ACERDataset
        self._train_dataset = None
        self._val_dataset = None

    # ---------------------
    # Class Constructor
    # ---------------------
    def to_dict(self) -> dict:
        """
        Sérialise les paramètres du DataModule dans un dictionnaire.
        Returns:
            dict: Un dictionnaire contenant les paramètres du DataModule.
        """
        return {
            "n_lookback_days": self.n_lookback_days,
            "horizon": self.horizon,
            "start_padding_days": self.start_padding_days,
            "end_padding_days": self.end_padding_days,
            "norm_by_dayly_deviation": self.norm_by_dayly_deviation,
            "sample_hour": self.sample_hour,
            "augment_data": self.augment_data,
            "year_no_augment": self.year_no_augment,
            "val_split": self._val_split,
            "normalize_volume": self.normalize_volume,
            "normalize_temperature": self.normalize_temperature,
            "scaling_mode_robust": self.scaling_mode_robust,
            "scaling_per_feature": self.scaling_per_feature,
            "return_fields": self.return_fields,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict, ACER_csv_path: str = "./Data/ACER_2000-2024.csv") -> 'ACERDataModule':
        """
        Crée un ACERDataModule à partir d'un dictionnaire de configuration.
        Args:
            config_dict (dict): Dictionnaire contenant les paramètres du DataModule.
            ACER_csv_path (str): Chemin vers le fichier CSV des données ACER.
        Returns:
            ACERDataModule: Un objet ACERDataModule configuré avec les paramètres du dictionnaire.
        """
        return cls(
            ACER_csv_path=ACER_csv_path,
            n_lookback_days=config_dict.get("n_lookback_days", 3),
            horizon=config_dict.get("horizon", 1),
            start_padding_days=config_dict.get("start_padding_days", 2),
            end_padding_days=config_dict.get("end_padding_days", 0),
            norm_by_dayly_deviation=config_dict.get("norm_by_dayly_deviation", False),
            sample_hour=config_dict.get("sample_hour", 23),
            augment_data=config_dict.get("augment_data", False),
            year_no_augment=config_dict.get("year_no_augment", []),
            val_split=config_dict.get("val_split", 0.2),
            normalize_volume=config_dict.get("normalize_volume", False),
            normalize_temperature=config_dict.get("normalize_temperature", False),
            scaling_mode_robust=config_dict.get("scaling_mode_robust", False),
            scaling_per_feature=config_dict.get("scaling_per_feature", False),
            return_fields=config_dict.get("return_fields", None),
        )

    # ---------------------
    # Data Module Methods
    # ---------------------
    def __invalidate_cache(self):
        self._cached_code += 1
    
    def is_ready(self) -> bool:
        '''
        Vérifie si le DataModule est prêt à être utilisé (séquence générée, données divisées et normalisation calculée).
        Returns:
            bool: True si prêt, False sinon.
        '''
        return self._allData_upToDate and self._split_upToDate and self._fitted

    def set_params(
        self,
        n_lookback_days: int | None = None,
        horizon: int | None = None,
        start_padding_days: int | None = None,
        end_padding_days: int | None = None,
        norm_by_dayly_deviation: bool | None = None,
        sample_hour: int | None = None,
        augment_data: bool | None = None,
        year_no_augment: list[int] | None = None,
        val_split: float | list[int] | None = None,
        normalize_volume: bool | None = None,
        normalize_temperature: bool | None = None,
        scaling_mode_robust: bool | None = None,
        scaling_per_feature: bool | None = None,
        return_fields: list[str] | None = None,
        take_only_first_system: bool | None = None
    ):
        '''
        Définit plusieurs paramètres du DataModule en une seule fois.
        Args:
            n_lookback_days (int | None): Nombre de jours précédents à inclure dans chaque séquence. Si None, ne modifie pas la valeur actuelle.
            horizon (int | None): Nombre de jours dans le futur à prédire. Si None, ne modifie pas la valeur actuelle.
            start_padding_days (int | None): Nombre de jours de padding à ajouter au début des séquences. Si None, ne modifie pas la valeur actuelle.
            end_padding_days (int | None): Nombre de jours de padding à ajouter à la fin des séquences. Si None, ne modifie pas la valeur actuelle.
            norm_by_dayly_deviation (bool | None): Si True, normalise les volumes par la déviation journalière des années précédentes. Si None, ne modifie pas la valeur actuelle.
            sample_hour (int | None): Heure de la journée que les échantillons sont prises (approximatif). Si None, ne modifie pas la valeur actuelle.
            augment_data (bool | None): Si True, active l'augmentation de données. Si None, ne modifie pas la valeur actuelle.
            year_no_augment (list[int] | None): Liste des années à ne pas augmenter. Si None, ne modifie pas la valeur actuelle.
            val_split (float | list[int] | None): Si float, fraction des données à utiliser pour la validation. Si list[int], liste des années à utiliser pour la validation. Si None, utilise la valeur actuelle.
            normalize_volume (bool | None): Si True, normalise les valeurs de type volume. Si None, ne modifie pas la valeur actuelle.
            normalize_temperature (bool | None): Si True, normalise les valeurs de type température. Si None, ne modifie pas la valeur actuelle.
            scaling_mode_robust (bool | None): Si True, utilise une normalisation robuste (médiane et IQR). Si None, ne modifie pas la valeur actuelle.
            scaling_per_feature (bool | None): Si True, calcule les statistiques de normalisation par feature. Si None, ne modifie pas la valeur actuelle.
            return_fields (list[str] | None): Liste des champs à retourner. Si None, ne modifie pas la valeur actuelle.
            take_only_first_system (bool | None): Si True, ne prend en compte que le premier système de chaque année. Si None, ne modifie pas la valeur actuelle.
        '''
        self.set_data_parameters(
            n_lookback_days,
            horizon,
            start_padding_days,
            end_padding_days,
            norm_by_dayly_deviation,
            sample_hour,
            augment_data,
            year_no_augment,
            take_only_first_system
        )

        if val_split is not None:
            self._val_split = val_split
            self._split_upToDate = False
            
        self.set_normalization_flags(
            normalize_volume,
            normalize_temperature,
            scaling_mode_robust,
            scaling_per_feature,
        )

        if return_fields is not None:
            self.set_return_fields(return_fields)

    def get_ready(self):
        '''
        Prépare le DataModule en générant la séquence, en divisant les données et en calculant les statistiques de normalisation.
        '''
        self.generate_sequence()
        self.split()
        self.compute_normalization_stats()

    # ---------------------
    # Data Parameters and Generation
    # ---------------------
    def set_data_parameters(
        self,
        n_lookback_days: int | None = None,
        horizon: int | None = None,
        start_padding_days: int | None = None,
        end_padding_days: int | None = None,
        norm_by_dayly_deviation: bool | None = None,
        sample_hour: int | None = None,
        augment_data: bool | None = None,
        year_no_augment: list[int] | None = None,
        take_only_first_system: bool | None = None,
    ):
        '''
        Définit les paramètres de génération des données.
        Args:
            n_lookback_days (int | None): Nombre de jours précédents à inclure dans chaque séquence. Si None, ne modifie pas la valeur actuelle.
            horizon (int | None): Nombre de jours dans le futur à prédire. Si None, ne modifie pas la valeur actuelle.
            start_padding_days (int | None): Nombre de jours de padding à ajouter au début des séquences. Si None, ne modifie pas la valeur actuelle.
            end_padding_days (int | None): Nombre de jours de padding à ajouter à la fin des séquences. Si None, ne modifie pas la valeur actuelle.
            norm_by_dayly_deviation (bool | None): Si True, normalise les volumes par la déviation journalière des années précédentes. Si None, ne modifie pas la valeur actuelle.
            sample_hour (int | None): Heure de la journée que les échantillons sont prises (approximatif). Si None, ne modifie pas la valeur actuelle.
            augment_data (bool | None): Si True, active l'augmentation de données. Si None, ne modifie pas la valeur actuelle.
            year_no_augment (list[int] | None): Liste des années à ne pas augmenter. Si None, ne modifie pas la valeur actuelle.
            take_only_first_system (bool | None): Si True, ne prend en compte que le premier système de chaque année. Si None, ne modifie pas la valeur actuelle.
        '''
        info_change = False
        if n_lookback_days is not None:
            if n_lookback_days != self.n_lookback_days:
                self.n_lookback_days = n_lookback_days
                info_change = True
        if horizon is not None:
            if horizon != self.horizon:
                self.horizon = horizon
                info_change = True
        if start_padding_days is not None:
            if start_padding_days != self.start_padding_days:
                self.start_padding_days = start_padding_days
                info_change = True
        if end_padding_days is not None:
            if end_padding_days != self.end_padding_days:
                self.end_padding_days = end_padding_days
                info_change = True
        if norm_by_dayly_deviation is not None:
            if norm_by_dayly_deviation != self.norm_by_dayly_deviation:
                self.norm_by_dayly_deviation = norm_by_dayly_deviation
                info_change = True
        if sample_hour is not None:
            if sample_hour != self.sample_hour:
                self.sample_hour = sample_hour
                info_change = True
        if augment_data is not None:
            if augment_data != self.augment_data:
                self.augment_data = augment_data
                info_change = True
        if year_no_augment is not None:
            if year_no_augment != self.year_no_augment:
                self.year_no_augment = year_no_augment
                info_change = True
        if take_only_first_system is not None:
            if take_only_first_system != self.take_only_first_system:
                self.take_only_first_system = take_only_first_system
                info_change = True

        if info_change:
            self._allData_upToDate = False
            self.__invalidate_cache()

    def generate_sequence(self, verbose: bool = False):
        '''
        Génère la séquence de données ACER en fonction des paramètres actuels.
        Args:
            verbose (bool): Si True, affiche des informations de progression.
        '''
        if self._allData_upToDate and self.allData_df is not None:
            if verbose:
                print("All data is already up to date.")
            return
        
        self.allData_df = ACER_helper.prep_sequence(
            ACER_df = self.ACER_df,
            volume_column_name = 'volume entaille' if not self.norm_by_dayly_deviation else 'volume_norm',
            n_lookback_days = self.n_lookback_days,
            horizon = self.horizon,
            start_padding_days = self.start_padding_days,
            end_padding_days = self.end_padding_days,
            sample_hour = self.sample_hour,
            verbose = verbose,
            augment_data = self.augment_data,
            year_no_augment = self.year_no_augment,
            take_only_first_system = self.take_only_first_system
        )
        self._allData_upToDate = True
        self.length = self.allData_df.shape[0]
        self._split_upToDate = False
        self.__invalidate_cache()

    # ---------------------
    # Split
    # ---------------------
    def set_val_split(self, val_split: float | list[int]):
        '''
        Définit la fraction des données à utiliser pour la validation ou les années à utiliser pour la validation.
        Args:
            val_split (float | list[int]): Si float, fraction des données à utiliser pour la validation. Si list[int], liste des années à utiliser pour la validation.
        '''
        if val_split != self._val_split:
            self._val_split = val_split
            self._split_upToDate = False
            self.__invalidate_cache()

    def split(self, val_split: float | list[int] | None = None):
        '''
        Divise les données en ensembles d'entraînement et de validation.
        Args:
            val_split (float | list[int] | None): Si float, fraction des données à utiliser pour la validation. Si list[int], liste des années à utiliser pour la validation. Si None, utilise la valeur actuelle.
        '''
        if val_split is not None:
            self.set_val_split(val_split)

        assert self._allData_upToDate, "Call generate_sequence() first"

        if self._split_upToDate:
            return
        
        idx = torch.arange(self.length)     
        if isinstance(self._val_split, float):
            n_val = int(self.length * self._val_split)
            self._train_idx = idx[:-n_val]
            self._val_idx = idx[-n_val:]
        elif isinstance(self._val_split, list) and len(self._val_split) != self.length:
            mask = self.allData_df.index.get_level_values('date').year.isin(self._val_split)
            self._val_idx = idx[mask]
            self._train_idx = idx[~mask]
        else:
            raise ValueError("val_split must be a float or a list of years")
        self._split_upToDate = True
        self.__invalidate_cache()
        
        if self._train_dataset is not None:
            self._train_dataset._idx = self._train_idx
        if self._val_dataset is not None:
            self._val_dataset._idx = self._val_idx
        
    # ---------------------
    # Normalization
    # ---------------------
    def set_normalization_flags(
            self,
            normalize_volume: bool | None = None,
            normalize_temperature: bool | None = None,
            scaling_mode_robust: bool | None = None,
            scaling_per_feature: bool | None = None
        ):
        """
        Définit les flags de normalisation et calcule les statistiques de normalisation.
        Args:
            normalize_volume (bool | None): Si True, normalise les valeurs de type volume. Si None, ne modifie pas la valeur actuelle.
            normalize_temperature (bool | None): Si True, normalise les valeurs de type température. Si None, ne modifie pas la valeur actuelle.
            scaling_mode_robust (bool | None): Si True, utilise une normalisation robuste (médiane et IQR). Si None, ne modifie pas la valeur actuelle.
            scaling_per_feature (bool | None): Si True, calcule les statistiques de normalisation par feature. Si None, ne modifie pas la valeur actuelle.
        """
        info_change = False
        if normalize_volume is not None:
            if normalize_volume != self.normalize_volume:
                self.normalize_volume = normalize_volume
                info_change = True
        if normalize_temperature is not None:
            if normalize_temperature != self.normalize_temperature:
                self.normalize_temperature = normalize_temperature
                info_change = True
        if scaling_mode_robust is not None:
            if scaling_mode_robust != self.scaling_mode_robust:
                self.scaling_mode_robust = scaling_mode_robust
                info_change = True
                self._fitted = False
        if scaling_per_feature is not None:
            if scaling_per_feature != self.scaling_per_feature:
                self.scaling_per_feature = scaling_per_feature
                info_change = True
                self._fitted = False

        if info_change:
            self.__invalidate_cache()

    def compute_normalization_stats(self):
        """
        Calcule les statistiques de normalisation pour les entrées de volume et de température.
        Si le flag de normalisation robuste est activé, utilise la médiane et l'IQR. Sinon, utilise la moyenne et l'écart-type.
        Si le flag de normalisation par feature est activé, calcule les statistiques par feature. Sinon, calcule globalement.
        """
        assert self._train_idx is not None or self._split_upToDate, "Call split() first"

        if self._fitted:
            return
        
        #  !!! Inutile, prendre que les 24 dernier heures dans tous les cas
        # if self.n_lookback_days >= 1:
        #     volume_tensor = ACER_helper.get_input_volume_lagged(self.allData_df, tensorDType=None)
        #     temperature_tensor = ACER_helper.get_input_temperature_lagged(self.allData_df, tensorDType=None)
        # else:
        #     volume_tensor = ACER_helper.get_target_volume(self.allData_df, tensorDType=None)
        #     temperature_tensor = ACER_helper.get_input_temperature_lagged(self.allData_df, tensorDType=None)

        self.normalization_stats['volume'] = NormHelper.compute_normalization_stats(
            ACER_helper.get_input_volume_lagged(self.allData_df, tensorDType=None)[self._train_idx][:,-1:],
            self.scaling_mode_robust,
            self.scaling_per_feature
        )
        self.normalization_stats['temperature'] = NormHelper.compute_normalization_stats(
            ACER_helper.get_input_temperature_lagged(self.allData_df, tensorDType=None)[self._train_idx][:,-24:],
            self.scaling_mode_robust,
            self.scaling_per_feature
        )
        for weather_var in ACER_helper.WEATHER_VARIABLES.values():
            self.normalization_stats["wv_" + weather_var] = NormHelper.compute_normalization_stats(
                ACER_helper.get_input_regex(self.allData_df, f"^T-1_{weather_var}-?[0-9]+$", tensorDType=None)[self._train_idx],
                self.scaling_mode_robust,
                self.scaling_per_feature
            )

        # print("Normalization stats computed:")
        # for key, stats in self.normalization_stats.items():
        #     print(f"  {key}:")
        #     for stat_name, value in stats.items():
        #         print(f"    {stat_name}: {value}")

        self._fitted = True
        self.__invalidate_cache()

    # ---------------------
    # Return Fields
    # ---------------------
    def set_return_fields(self, return_fields: list[str] | None = None):
        """
        Définit les champs à retourner dans '__getitem__' des datasets.
        Args:
            return_fields (list[str] | None): Liste des champs à retourner. Si None, utilise les champs par défaut.
        """
        
        if self.return_fields == return_fields:
            return
        
        if return_fields is None:
            # Defaut retourne la concatenation de input_temperature_lagged et input_volume_lagged, suivi de target_volume
            self.return_fields = [
                "concat[input_temperature_lagged_view[-1,3,24], input_volume_lagged_view[-1,3,1]]",
                "target_volume",
            ]
        else:
            def replace_expr(match):
                return str(eval(match.group(1), {"__builtins__": {}}, vars(self)))
            
            _processed_fields = []
            for field in return_fields:
                _processed_fields.append(
                    re.sub(
                        r"\$\{([^}]+)\}",
                        replace_expr,
                        field
                    )
                )

            self.return_fields = _processed_fields

        self.__invalidate_cache()

    def get_tensor(self, field: str, idx: torch.Tensor, tensorDType: torch.dtype | None = None) -> torch.Tensor:
        """
        Récupère un tenseur spécifique pour les indices donnés.
        Args:
            field (str): Le champ à récupérer.
            idx (torch.Tensor): Les indices des données à récupérer.
            tensorDType (torch.dtype | None): Le dtype du tenseur retourné. Si None, garde le dtype original.
        Returns:
            torch.Tensor: Le tenseur correspondant au champ et aux indices spécifiés.
        """
        dm = self

        # -- Empty Tensor Shape (idx.size(0), 0)
        if "empty" in field:
            t = torch.empty((idx.size(0), 0), dtype=tensorDType)

        # -- Input Volume Lagged --
        elif field.startswith("input_volume_lagged"):
            t = ACER_helper.get_input_volume_lagged(dm.allData_df, tensorDType=tensorDType)[idx]
            if dm.normalize_volume and dm._fitted:
                t = NormHelper.apply_normalization(
                    t,
                    dm.normalization_stats['volume']
                )
        
        # -- Input Temperature Lagged --
        elif field.startswith("input_temperature_lagged"):
            t = ACER_helper.get_input_temperature_lagged(dm.allData_df, tensorDType=tensorDType)[idx]
            if dm.normalize_temperature and dm._fitted:
                t = NormHelper.apply_normalization(
                    t,
                    dm.normalization_stats['temperature']
                )

        # -- Input Temperature Forecast --
        elif field.startswith("input_temperature_forecast"):
            t = ACER_helper.get_input_temperature_forecast(dm.allData_df, tensorDType=tensorDType)[idx]
            if dm.normalize_temperature and dm._fitted:
                t = NormHelper.apply_normalization(
                    t,
                    dm.normalization_stats['temperature']
                )

        # -- Target Volume --
        elif field.startswith("target_volume"):
            t = ACER_helper.get_target_volume(dm.allData_df, tensorDType=tensorDType)[idx]
            if dm.normalize_volume and dm._fitted:
                t = NormHelper.apply_normalization(
                    t,
                    dm.normalization_stats['volume']
                )

        # -- Autre variable météo --
        elif field.startswith("weather_lagged") or field.startswith("weather_forecast"):
            type_part = field[field.find("_")+1:field.find("(")]  # "lagged" ou "forecast"
            # Extraire la partie entre paranthèses
            start_idx = field.find("(")
            end_idx = field.find(")")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                weather_var = field[start_idx+1:end_idx]
                weather_var = ACER_helper.WEATHER_VARIABLES.get(weather_var, weather_var)  # Permet d'utiliser des alias définis dans ACER_helper
                regex_pattern = f"^T{'-'if type_part == 'lagged' else ''}[0-9]+_{weather_var}-?[0-9]+$"
                t = ACER_helper.get_input_regex(dm.allData_df, regex_pattern, tensorDType=tensorDType)[idx]

                if dm.normalize_temperature and dm._fitted and weather_var in ACER_helper.WEATHER_VARIABLES.values():
                    norm_key = "wv_" + weather_var
                    if norm_key in dm.normalization_stats:
                        t = NormHelper.apply_normalization(
                            t,
                            dm.normalization_stats[norm_key]
                        )
                    else:
                        raise ValueError(f"No normalization stats found for weather variable: {weather_var}")
            else:
                raise ValueError(f"Invalid weather variable specification in field: {field}")
            
        # -- Degres Jour --
        elif field.startswith("dj_0_sum_lagged"):
            regex_pattern = f"^T-[0-9]+_(dj_0_sum)$"
            t = ACER_helper.get_input_regex(dm.allData_df, regex_pattern, tensorDType=tensorDType)[idx]

        elif field.startswith("dj_0_sum_forecast"):
            regex_pattern = f"^T[0-9]+_(dj_0_sum)$"
            t = ACER_helper.get_input_regex(dm.allData_df, regex_pattern, tensorDType=tensorDType)[idx]

        elif field.startswith("regex"):
            # Extraire la partie entre paranthèses
            start_idx = field.find(":")
            end_idx = field.find("::")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                regex_pattern = field[start_idx+1:end_idx]
                t = ACER_helper.get_input_regex(dm.allData_df, regex_pattern, tensorDType=tensorDType)[idx]
            else:
                raise ValueError(f"Invalid regex specification in field: {field}")
        # -- Volume Journalier Moyen Année Précédente --
        elif field.startswith("volume_journalier_moyen"):
            id_date_system = ACER_helper.get_id_date_system(dm.allData_df, dm.past_years_df)
            t = ACER_helper.get_volume_journalier_moyen(id_date_system, tensorDType=tensorDType)[idx]

        # -- ID Date System / Meta--
        elif field.startswith("id_date_system") or field.startswith("meta"):
            id_date_system = ACER_helper.get_id_date_system(dm.allData_df, dm.past_years_df)
            # Retourne un tensor d'objets (strings/dates)
            # t = torch.tensor(
            #     id_date_system.iloc[idx].to_numpy(),
            #     # dtype=torch.object
            # )
            return id_date_system.iloc[idx].to_numpy()

        # -- Field inconnu --
        else:
            raise ValueError(f"Unknown field: {field}")
        
        # -- Modificateur de vue (au besoin) --
        if "_view" in field:
            # Extrait la vue
            view_part = field.split("_view")[1]
            if view_part.startswith("[") and view_part.endswith("]"):
                indices = [int(i) for i in view_part[1:-1].split(",")]
                t = t.view(*indices)
            else:
                raise ValueError(f"Invalid view specification in field: {field}")

        return t

    # ---------------------
    # Helper Dataset
    # ---------------------
    def get_years(self) -> list[int]:
        """
        Récupère la liste des années présentes dans les données.
        Returns:
            list[int]: Liste des années uniques présentes dans les données.
        """
        return self.allData_df.index.get_level_values('date').year.unique().tolist()

    def gen_train_dataset(self) -> 'ACERDataset':
        assert self._train_idx is not None, "Call split() first"
        if self._train_dataset is None:
            self._train_dataset = ACERDatasetAutoIdx(self, ACERDatasetAutoIdx.AutoIdxType.Train)
        return self._train_dataset
    
    def gen_val_dataset(self) -> 'ACERDataset':
        assert self._val_idx is not None, "Call split() first"
        if self._val_dataset is None:
            self._val_dataset = ACERDatasetAutoIdx(self, ACERDatasetAutoIdx.AutoIdxType.Val)
        return self._val_dataset
    
    def get_train_val_datasets(self) -> tuple['ACERDataset', 'ACERDataset']:
        return (
            self.gen_train_dataset(),
            self.gen_val_dataset()
        )

    def gen_full_dataset(self) -> 'ACERDataset':
        full_idx = torch.arange(self.length)
        return ACERDataset(self, full_idx)

    def get_first_system_of_year_idx(self, year: int, first_system: str = None) -> torch.Tensor:
        mask = self.allData_df.index.get_level_values('date').year == year
        if first_system is None:
            first_system = self.allData_df[mask].index.get_level_values('system').unique()[0]
            # print(f"First system for year {year} is {first_system}")
        mask = (self.allData_df.index.get_level_values('date').year == year) & (self.allData_df.index.get_level_values('system') == first_system)
        idx = torch.arange(self.length)[mask]
        return idx
    
    def get_first_system_of_year_dataset(self, year: int, first_system: str = None) -> 'ACERDataset':
        # mask = self.allData_df.index.get_level_values('date').year == year
        # if first_system is None:
        #     first_system = self.allData_df[mask].index.get_level_values('system').unique()[0]
        #     print(f"First system for year {year} is {first_system}")
        # mask = (self.allData_df.index.get_level_values('date').year == year) & (self.allData_df.index.get_level_values('system') == first_system)
        # idx = torch.arange(self.length)[mask]
        idx = self.get_first_system_of_year_idx(year, first_system)
        return ACERDataset(self, idx)

    def get_first_system_for_years(self, years: list[int]) -> 'ACERDataset':
        import pandas as pd
        dates = self.allData_df.index.get_level_values("date")
        systems = self.allData_df.index.get_level_values("system")
    
        mask = pd.Series(False, index=self.allData_df.index)
    
        for year in years:
            year_mask = dates.year == year

            # Skip years with no data
            if not year_mask.any():
                continue

            first_system = systems[year_mask][0]
            mask |= year_mask & (systems == first_system)
    
        idx = torch.arange(len(self.allData_df))[torch.tensor(mask.to_list())]
        return ACERDataset(self, idx)

from torch.utils.data import Dataset
class ACERDataset(Dataset):
    def __init__(self, data_module: ACERDataModule, indices: torch.Tensor):
        self._dm = data_module
        self._idx = indices

        self._cached_code = -1
        self._cached_items = None
        self._cached_device = None
        self._cached_dtype = None
        self._cached_norm_volume_center = None
        self._cached_norm_volume_scale = None

    # ---------------------
    # Internal resolution
    # ---------------------
    def _resolve_field(self, field: str) -> torch.Tensor:
        return self._dm.get_tensor(
            field,
            self._idx,
            tensorDType=None
        )

    
    # ---------------------
    # GPU precache (batché)
    # ---------------------
    def is_precached_valid(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> bool:
        """
        Vérifie si le cache actuel est valide pour le device et le dtype spécifiés.
        Args:
            device (torch.device | str): Le device à vérifier (ex: 'cuda', 'cpu').
            dtype (torch.dtype | None): Le dtype à vérifier (ex: torch.float16, torch.bfloat16). Si None, vérifie uniquement le device.
        Returns:
            bool: True si le cache est valide, False sinon.
        """
        return (
            self._dm._cached_code == self._cached_code
            and self._dm.is_ready()
            and self._cached_items is not None
            and self._cached_device == device
            and self._cached_dtype == dtype
        )
    
    def ensure_precached(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ):
        """
        Assure que le cache est valide pour le device et le dtype spécifiés.
        Si le cache n'est pas valide, appelle 'precache_to_device' pour le mettre à jour.
        Args:
            device (torch.device | str): Le device à vérifier (ex: 'cuda', 'cpu').
            dtype (torch.dtype | None): Le dtype à vérifier (ex: torch.float16, torch.bfloat16). Si None, vérifie uniquement le device.
        """
        if not self.is_precached_valid(device, dtype):
            self.precache_to_device(device, dtype)

    def precache_to_device(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ):
        """
        Precaclcule et déplace tous les tenseurs pour '__getitem__' vers le device spécifié et le dtype spécifié.
        Les tenseurs deplacés sont ceux spécifiés dans 'getitem_return_fields'.
        La normalisation est appliquée selon les flags de normalisation.
        !!! Le CPU reste inchangé !!!
        Args:
            device (torch.device | str): Le device vers lequel déplacer les tenseurs (ex: 'cuda', 'cpu').
            dtype (torch.dtype | None): Le dtype vers lequel caster les tenseurs (ex: torch.float16, torch.bfloat16). Si None, garde le dtype original.
        """
        assert self._dm.is_ready(), "DataModule is not ready. Call get_ready() first."

        cached = []

        with torch.no_grad():
            for field in self._dm.return_fields:

                # --- Résolution de concatenation de champs
                if field.startswith("concat["):
                    names = field[len("concat["):-1].split(", ")
                    tensors = [self._resolve_field(n) for n in names]
                    batch = torch.cat(tensors, dim=-1)

                # --- Résolution de champ simple
                else:
                    batch = self._resolve_field(field)

                # --- Déplacement vers le device et conversion de dtype
                batch = batch.to(device)
                if dtype is not None:
                    batch = batch.to(dtype)

                cached.append(batch)

        self._cached_code = self._dm._cached_code
        self._cached_items = tuple(cached)
        self._cached_device = device
        self._cached_dtype = dtype
        self._cached_norm_volume_center = self._dm.normalization_stats['volume']['center'].to(device) if self._dm.normalize_volume else 0
        self._cached_norm_volume_scale = self._dm.normalization_stats['volume']['scale'].to(device) if self._dm.normalize_volume else 1

    def inverse_normalize_volume(self, volume_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse la normalisation des tenseurs de volume.
        Args:
            volume_tensor (torch.Tensor): Le tenseur de volume normalisé à inverser.
        Returns:
            torch.Tensor: Le tenseur de volume original avant normalisation.
        """
        return volume_tensor * self._cached_norm_volume_scale + self._cached_norm_volume_center

    # ---------------------
    # Dataset API
    # ---------------------
    def __len__(self):
        return self._idx.shape[0]

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self._cached_items)
    
class ACERDatasetAutoIdx(ACERDataset):
    class AutoIdxType(Enum):
        Train = 1
        Val = 2

    def __init__(self, data_module: ACERDataModule, idx_type: AutoIdxType):
        if idx_type == self.AutoIdxType.Train:
            indices = data_module._train_idx
        elif idx_type == self.AutoIdxType.Val:
            indices = data_module._val_idx
        else:
            raise ValueError(f"Unknown AutoIdxType: {idx_type}")
        
        super().__init__(data_module, indices)