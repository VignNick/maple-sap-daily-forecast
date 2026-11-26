from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from .meteo_db import MeteoDB

modified_volume_col_name = 'volume_mod'
# Correspondance entre variable météo et nom de colonne dans la df pivotée
WEATHER_VARIABLES = {
    "temperature_2m": "Temperature_Hour",
    "apparent_temperature": "AppTemperature_Hour",
    "shortwave_radiation": "Radiation_Hour",
    "precipitation": "Precipitation_Hour",
    "cloud_cover": "Cloud_Cover_Hour",
    "wind_speed_10m": "Wind_Speed_Hour",
    "vapour_pressure_deficit": "VPD_Hour"
}

def compute_past_years_weighted_value(
    df,
    date_col='date',
    system_col='system',
    value_col='volume entaille',
    weights=np.array([0.2, 0.3, 0.5])
):
    """
    Calcule une valeur annuelle pondérée basée uniquement sur les années passées.

    Étapes :
    1. Somme annuelle par (année, système)
    2. Moyenne annuelle tous systèmes confondus
    3. Calcul du nombre max de jours non nuls par année (nombre de jours de coulée)
    4. Calcul de la moyenne journalière ajustée par ce nombre de jours
    5. Moyenne pondérée des années passées (logique métier spécifique)

    Returns
    -------
    DataFrame avec :
    - annee
    - volume saisonnier moyen (litre/entaille)
    - nb_jours_en_coulee
    - volume_journalier_moyen (litre/entaille)
    - volume_journalier_moyen_annee_precedente (litre/entaille)
    """

    # ---- 1et2. Valeur annuelle moyenne par système
    yearly = (
        df
        .assign(annee=df[date_col].dt.year)
        .groupby(['annee', system_col], as_index=False)[value_col]
        .sum()
        .groupby('annee', as_index=False)[value_col]
        .mean()
        .sort_values('annee')
        .rename(columns={value_col: 'volume saisonnier moyen (litre/entaille)'})
    )

    # ---- 3. Max jours non nuls par année
    max_jours_par_annee = (
        df
        .assign(annee=df[date_col].dt.year)
        .groupby(['annee', system_col])[value_col]
        .agg(nb_jours_en_coulee=lambda x: ((x != 0) & x.notna()).sum())
        .reset_index()
        .groupby('annee', as_index=False)['nb_jours_en_coulee']
        .max()
    )

    # ---- 3.1. Merge
    yearly = yearly.merge(max_jours_par_annee, on='annee', how='left')

    # ---- 4. Moyenne journalière ajustée
    yearly['volume_journalier_moyen (litre/entaille)'] = np.where(
        yearly['nb_jours_en_coulee'] > 0,
        yearly['volume saisonnier moyen (litre/entaille)'] / yearly['nb_jours_en_coulee'],
        np.nan
    )

    # ---- 5. Moyenne pondérée des années passées
    values = yearly['volume_journalier_moyen (litre/entaille)'].values
    result = []

    for i in range(len(values)):
        if i == 0:
            result.append(values[i])
            continue

        j = min(i, len(weights))
        w = weights[-j:]
        w = w / w.sum()

        result.append(np.dot(values[i - j:i], w))

    yearly['volume_journalier_moyen_annee_precedente (litre/entaille)'] = result

    return yearly.reset_index(drop=True)

def get_ACER_df(ACER_csv_path: str) -> pd.DataFrame:
    """
    Charge les données ACER depuis le fichier CSV et retourne un DataFrame pandas.

    Args:
        ACER_csv_path (str): Chemin vers le fichier CSV des données ACER.
    Returns:
        pd.DataFrame: DataFrame contenant les données ACER avec les colonnes 'date', 'system', et 'volume entaille'.
    """
    # Charger les données ACER depuis le fichier CSV
    ACER_df = pd.read_csv(ACER_csv_path, parse_dates=['date'])
    ACER_df.sort_values(by='date', inplace=True)
    ACER_df.reset_index(drop=True, inplace=True)
    # Remplacer nan systèmes par 'Unknown'
    ACER_df['system'] = ACER_df['system'].fillna('Unknown')
    # Garder uniquement les colonnes nécessaires
    ACER_df = ACER_df[['date', 'system', 'volume entaille']]

    # Drop les ligne pour annee 2022 et de systeme 'A', 'B', 'C' (données incomplètes)
    ACER_df = ACER_df[~((ACER_df['date'].dt.year == 2022) & (ACER_df['system'].isin(['A', 'B', 'C'])))]

    return ACER_df

def ACER_df_append_past_years_norm(ACER_df: pd.DataFrame, weights=np.array([0.2, 0.3, 0.5])) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ajoute une colonne de volume normalisé basé sur la moyenne journalière pondérée des années passées.

    Args:
        ACER_df (pd.DataFrame): DataFrame contenant les données ACER avec les colonnes 'date', 'system', et 'volume entaille'.
        weights (np.array): Poids pour le calcul de la moyenne pondérée des années passées.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame ACER avec une colonne supplémentaire 'volume_norm'.
            - DataFrame des valeurs annuelles pondérées calculées.
    """
    past_years_df = compute_past_years_weighted_value(df=ACER_df, weights=weights)
    ACER_df = ACER_df.copy()
    ACER_df['volume_norm'] = ACER_df['volume entaille'] / ACER_df['date'].dt.year.map(
               past_years_df.set_index('annee')['volume_journalier_moyen_annee_precedente (litre/entaille)']
           )
    return ACER_df, past_years_df

def get_ACERdf_PastYearsDf(ACER_csv_path: str, weights=np.array([0.2, 0.3, 0.5])) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les données ACER depuis le fichier CSV et ajoute une colonne de volume normalisé basé sur la moyenne journalière pondérée des années passées.

    Args:
        ACER_csv_path (str): Chemin vers le fichier CSV des données ACER.
        weights (np.array): Poids pour le calcul de la moyenne pondérée des années passées.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame ACER avec une colonne supplémentaire 'volume_norm'.
            - DataFrame des valeurs annuelles pondérées calculées.
    """
    ACER_df = get_ACER_df(ACER_csv_path)
    ACER_df, past_years_df = ACER_df_append_past_years_norm(ACER_df, weights)
    return ACER_df, past_years_df

def get_meteo_for_dates(meteo_db: 'MeteoDB', start_date: pd.Timestamp, end_date: pd.Timestamp, latitude: float, longitude: float, sample_hour: int = 23, variable_list: list[str] = ["temperature_2m"], columns_name_list: list[str] | None = None, alternate_var: bool = True) -> pd.DataFrame:
    """
    Récupère les données météo pour une plage de dates spécifiée.

    Args:
        start_date (pd.Timestamp): Date de début.
        end_date (pd.Timestamp): Date de fin.
        latitude (float): Latitude du lieu.
        longitude (float): Longitude du lieu.
        sample_hour (int): L'heure à laquelle les échantillons de volume sont pris (0-23).
        variable_list (list[str]): Liste des variables météo à récupérer.
        columns_name_list (list[str] | None): Liste des noms de colonnes à utiliser pour renommer les variables météo. Si None, utilise variable_list.
        alternate_var (bool): Si True, alterne les variables pour chaque heure dans le DataFrame pivoté.
    Returns:
        pd.DataFrame: DataFrame contenant les données météo avec les colonnes 'date' et les variables spécifiées.
    """
    hour_shift = 23 - sample_hour  # Ajuster si nécessaire

    start_date = start_date.replace(hour=0, minute=0, second=0) - pd.Timedelta(hours=hour_shift)
    end_date = end_date.replace(hour=23, minute=0, second=0) - pd.Timedelta(hours=hour_shift)

    temp_24h = meteo_db.load_hourly(
        latitude=latitude,
        longitude=longitude,
        start=start_date.tz_localize("America/Montreal", nonexistent=timedelta(hours=-1)),
        end=end_date.tz_localize("America/Montreal", nonexistent=timedelta(hours=1)),
        variable_names=variable_list
    )

    if columns_name_list is None:
        columns_name_list = variable_list
    else:
        assert len(columns_name_list) == len(variable_list), "columns_name_list must have the same length as variable_list"
        # Renommer les colonnes si nécessaire
        temp_24h.rename(columns=dict(zip(variable_list, columns_name_list)), inplace=True)

    # Grouper par jour (1h par colonne) la df est timestamp, temperature_2m
    temp_24h['date'] = temp_24h['timestamp'].dt.date
    temp_24h.loc[temp_24h['timestamp'].dt.hour > sample_hour, 'date'] = temp_24h['timestamp'].dt.date + pd.Timedelta(days=1)
    temp_24h['hour'] = temp_24h['timestamp'].dt.hour
    temp_24h.loc[temp_24h['hour'] > sample_hour, 'hour'] -= 24

    if alternate_var:
        new_columns = pd.MultiIndex.from_tuples(
            [(columns_name_list[i % len(columns_name_list)], hour) for hour in range(-hour_shift, 24-hour_shift) for i in range(len(columns_name_list))],
        names=['variable', 'hour']
        )
    else:
        new_columns = pd.MultiIndex.from_product(
            [columns_name_list, range(-hour_shift, 24-hour_shift)],
            names=['variable', 'hour']
        )

    pivoted_temp = (
        temp_24h
        .pivot(index='date', columns='hour', values=columns_name_list)
        .reindex(columns=new_columns)  # Assurer que toutes les variables et heures sont présentes
        .interpolate(method='linear', axis=1, limit_direction='both')  # Interpoler les valeurs manquantes
    )

    # Aplatir les noms de colonnes MultiIndex
    pivoted_temp.columns = [f"{var}{hour}" for var, hour in pivoted_temp.columns]

    # Set index as a pandas DateTimeIndex
    pivoted_temp.index = pd.to_datetime(pivoted_temp.index)
    return pivoted_temp

def gaussian_smooth_1d(x, sigma=2, kernel_size=9):
    if len(x) < kernel_size:
        kernel_size = len(x) if len(x) % 2 == 1 else len(x) - 1

    if kernel_size < 3:
        return x  # Pas de lissage possible pour des séquences très courtes

    # Creer un noyau gaussien 1D
    radius = kernel_size // 2
    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Appliquer la convolution pour lisser les données
    return np.convolve(x, kernel, mode='same')

def augment_row(row, columns_prefix='Temperature_Hour', dayly_shift_range=(-2, 2), scaling_range=(0.95, 1.05), noise_std_factor=0.02, noise_smoothing_sigma=2):
    selected_cols = [col for col in row.index if col.startswith(columns_prefix)]
    tmp = row[selected_cols].values.copy()

    base_std = np.std(tmp)

    # Daily shift (most important)
    if dayly_shift_range is not None:
        shift = np.random.uniform(*dayly_shift_range)
        tmp += shift

    # Scaling (moderate effect)
    if scaling_range is not None:
        scale = np.random.uniform(*scaling_range)
        tmp *= scale

    # Smooth noise (small!)
    if noise_std_factor is not None:
        noise = np.random.normal(0, noise_std_factor * base_std, size=tmp.shape)
        noise = gaussian_smooth_1d(noise, sigma=noise_smoothing_sigma)
        tmp += noise

    tmp = gaussian_smooth_1d(tmp, sigma=1)

    row[selected_cols] = tmp
    return row

def prep_sequence(ACER_df: pd.DataFrame, volume_column_name: str, n_lookback_days: int = 3, horizon: int = 1, start_padding_days: int = 2, end_padding_days: int = 0, verbose: bool = False, sample_hour: int = 23, augment_data: bool = False, year_no_augment: list[int] = [], take_only_first_system: bool = False) -> pd.DataFrame:
    """
    Prépare les séquences de données ACER en ajoutant du padding au début et à la fin.
    Ajoute également la météo de la journée.

    Args:
        ACER_df (pd.DataFrame): DataFrame contenant les données ACER avec les colonnes 'date', 'system', et le nom de la colonne de volume spécifié.
        volume_column_name (str): Nom de la colonne de volume à utiliser.
        n_lookback_days (int): Nombre de jours précédents à inclure dans chaque séquence.
        horizon (int): Nombre de jours dans le futur à prédire.
        start_padding_days (int): Nombre de jours de padding à ajouter au début des séquences.
        end_padding_days (int): Nombre de jours de padding à ajouter à la fin des séquences.
        verbose (bool): Si True, affiche des informations de progression.
        sample_hour (int): L'heure à laquelle les échantillons de volume sont pris (0-23).
        augment_data (bool): Si True, augmente les données.
        year_no_augment (list[int]): Liste des années pour lesquelles ne pas augmenter les données.
        take_only_first_system (bool): Si True, ne prend en compte que le premier système de chaque année.
    Returns:
        pd.DataFrame: DataFrame contenant les séquences de données préparées avec les colonnes lagged et horizon.
    """
    meteo = MeteoDB()
    latitude = 46.0777 #latitude du centre ACER
    longitude = -71.8853 #longitude du centre ACER
    # variable_list = ["temperature_2m", "shortwave_radiation", "precipitation", "cloud_cover", "wind_speed_10m", "vapour_pressure_deficit"]
    # columns_list = [WEATHER_VARIABLES[var] for var in variable_list]

    variable_list = list(WEATHER_VARIABLES.keys())
    columns_list = list(WEATHER_VARIABLES.values())

    if horizon < 1:
        horizon = 1

    years = ACER_df['date'].dt.year.unique()
    allData = []

    for year in years:
        systems = ACER_df[ACER_df['date'].dt.year == year]['system'].unique()

        for system in systems:
            if verbose:
                print(f"Processing year {year} and system {system}", end="... \n\t\t")
            system_df = ACER_df[(ACER_df['date'].dt.year == year) & (ACER_df['system'] == system)][['date', 'system', volume_column_name]].copy()
            # Changer le nom de la colonne volume_column_name en 'volume_mod' pour plus de cohérence
            system_df.rename(columns={volume_column_name: modified_volume_col_name}, inplace=True)

            system_df.set_index('date', inplace=True)

            start_date = system_df.index.min() - pd.Timedelta(days=start_padding_days + n_lookback_days)
            end_date = (system_df.index.max() + pd.Timedelta(days=end_padding_days + horizon - 1)).replace(hour=23, minute=0, second=0) # end_date hour to 23:00:00

            full_date_range = pd.date_range(start=start_date,
                                            end=end_date,
                                            freq='D')

            system_df = system_df.reindex(full_date_range).fillna({
                modified_volume_col_name: 0,
                'system': system
            })
            system_df.index.name = 'date'


            pivoted_temp = get_meteo_for_dates(
                meteo_db=meteo,
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                sample_hour=sample_hour,
                variable_list=variable_list,
                columns_name_list=columns_list,
                alternate_var=False
            )

            merged_df = system_df.merge(pivoted_temp, left_index=True, right_index=True, how='left')
            merged_df.reset_index(inplace=True)
            merged_df.set_index(['date', 'system'], inplace=True)

            # Ajouter une colonne pour le numéro de jour dans la saison premiere valeur 1 pour le premier jour de la saison
            merged_df['season_day'] = merged_df.groupby('system').cumcount()-1
            season_length = merged_df.groupby('system')['season_day'].max().max()
            merged_df['season_position'] = merged_df['season_day'] / season_length

            merged_df['temp_min'] = merged_df.filter(regex='^Temperature_Hour-?[0-9]+$').min(axis=1)
            merged_df['temp_max'] = merged_df.filter(regex='^Temperature_Hour-?[0-9]+$').max(axis=1)
            merged_df['temp_mean'] = merged_df.filter(regex='^Temperature_Hour-?[0-9]+$').mean(axis=1)
            merged_df['dj_0'] = merged_df.apply(lambda row: max(0, (row['temp_min'] + row['temp_max']) / 2), axis=1)
            merged_df['dj_0_sum'] = merged_df['dj_0'].cumsum()
            merged_df['nb_hour_under_0'] = merged_df.filter(regex=r'^Temperature_Hour-?[0-9]+$').apply(lambda row: (row < 0).sum(), axis=1)
            merged_df['nb_hour_over_0'] = merged_df.filter(regex=r'^Temperature_Hour-?[0-9]+$').apply(lambda row: (row >= 0).sum(), axis=1)

            # Apparent temperature calculations
            merged_df['app_temp_min'] = merged_df.filter(regex='^AppTemperature_Hour-?[0-9]+$').min(axis=1)
            merged_df['app_temp_max'] = merged_df.filter(regex='^AppTemperature_Hour-?[0-9]+$').max(axis=1)
            merged_df['app_dj_0'] = merged_df.apply(lambda row: max(0, (row['app_temp_min'] + row['app_temp_max']) / 2), axis=1)
            merged_df['app_dj_0_sum'] = merged_df['app_dj_0'].cumsum()

            def lagDays(df):
                # Ajouter les colonnes pour les "Lagged Values" et les "Horizon Values" (Target)
                lagged_df = pd.concat(
                    [
                        df.shift(-lag).add_prefix(f"T{lag}_")
                        for lag in range(-n_lookback_days, horizon)
                    ],
                    axis=1
                ).dropna()

                lagged_df = lagged_df.merge(
                    ACER_df[['date', 'system', 'volume entaille']],
                    on=['date', 'system'],
                    how='left'
                ).fillna({'volume entaille': 0}).set_index(merged_df.index.names)

                return lagged_df

            allData.append(lagDays(merged_df))

            if augment_data and year not in year_no_augment:
                augmented_data = merged_df.copy().apply(augment_row, axis=1, columns_prefix='Temperature_Hour', dayly_shift_range=(-0.5, 0.5), scaling_range=(0.90, 1.10), noise_std_factor=0.08)
                augmented_data.reset_index(inplace=True)
                augmented_data['system'] = system+'_tempAug'
                augmented_data.set_index(['date', 'system'], inplace=True)
                allData.append(lagDays(augmented_data))

                augmented_data = merged_df.copy().apply(augment_row, axis=1, columns_prefix='Temperature_Hour', dayly_shift_range=(-0.5, 0.5), scaling_range=(0.90, 1.10), noise_std_factor=0.08)
                augmented_data = augmented_data.apply(augment_row, axis=1, columns_prefix='volume_mod', dayly_shift_range=None, scaling_range=(0.85, 1.10), noise_std_factor=None)
                augmented_data.reset_index(inplace=True)
                augmented_data['system'] = system+'_AllAug'
                augmented_data.set_index(['date', 'system'], inplace=True)
                allData.append(lagDays(augmented_data))
            
            if verbose:
                print(f"\t✓ done: nb_jours={len(merged_df)}  from {start_date.date()} to {end_date.date()}")

            if take_only_first_system:
                break  # Sortir de la boucle des systèmes après le premier
            
    return pd.concat(allData)

def get_id_date_system(allData_df: pd.DataFrame, past_years_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Extrait l'index (date, system) pour référence et ajoute des colonnes supplémentaires pour l'analyse post-traitement.

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        ACER_df (pd.DataFrame): DataFrame contenant les données ACER originales.
        past_years_df (pd.DataFrame | None): DataFrame des valeurs annuelles pondérées calculées. Si None, cette étape est ignorée.
    Returns:
        pd.DataFrame: DataFrame des identifiants de date et de système avec des colonnes supplémentaires.
    """
    id_date_system = allData_df.index.to_frame(index=False)
    # Y ajouter d'aure colonnes pour faciliter l'analyse post-traitement
    id_date_system['volume entaille'] = allData_df['volume entaille'].values
    id_date_system['volume_entrainement']= allData_df.filter(regex=f'^T0_{modified_volume_col_name}$').values

    if past_years_df is not None:
        id_date_system['volume_journalier_moyen_annee_precedente (litre/entaille)'] = id_date_system['date'].dt.year.map(
            past_years_df.set_index('annee')['volume_journalier_moyen_annee_precedente (litre/entaille)']
        )

    return id_date_system

def get_volume_journalier_moyen(id_date_system: pd.DataFrame, tensorDType : torch.dtype | None = None) -> torch.Tensor | None:
    """
    Extrait la colonne 'volume_journalier_moyen_annee_precedente (litre/entaille)' en tant que tenseur PyTorch.
    
    Args:
        id_date_system (pd.DataFrame): DataFrame des identifiants de date et de système.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur de la colonne 'volume_journalier_moyen_annee_precedente (litre/entaille)'.
    """
    if 'volume_journalier_moyen_annee_precedente (litre/entaille)' not in id_date_system.columns:
        return None
    return torch.tensor(
        id_date_system['volume_journalier_moyen_annee_precedente (litre/entaille)'].values,
        dtype=tensorDType
    ).unsqueeze(-1)



def get_input_volume_lagged(allData_df: pd.DataFrame, tensorDType : torch.dtype | None = None) -> torch.Tensor:
    """
    Extrait le tenseur d'entrée pour les volumes d'eau des jours précédents (lagged).

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur des volumes d'eau des jours précédents (lagged).
    """
    return torch.tensor(
        allData_df.filter(regex=f'^T-[0-9]+_{modified_volume_col_name}$').values,
        dtype=tensorDType
    )

def get_input_temperature_lagged(allData_df: pd.DataFrame, tensorDType : torch.dtype | None = None) -> torch.Tensor:
    """
    Extrait le tenseur d'entrée pour les températures observées des jours précédents (lagged).

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur des températures observées des jours précédents (lagged).
    """
    return torch.tensor(
        allData_df.filter(regex='^T-[0-9]+_Temperature_Hour-?[0-9]+$').values,
        dtype=tensorDType
    )

def get_input_temperature_forecast(allData_df: pd.DataFrame, tensorDType : torch.dtype | None = None) -> torch.Tensor:
    """
    Extrait le tenseur d'entrée pour les températures prévues pour les jours à venir (forecast/horizon).

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur des températures prévues pour les jours à venir (forecast/horizon).
    """
    return torch.tensor(
        allData_df.filter(regex='^T[0-9]+_Temperature_Hour-?[0-9]+$').values,
        dtype=tensorDType
    )

def get_input_regex(allData_df: pd.DataFrame, regex_pattern: str, tensorDType : torch.dtype | None = None) -> torch.Tensor:
    """
    Extrait un tenseur d'entrée basé sur un motif regex spécifique.

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        regex_pattern (str): Motif regex pour filtrer les colonnes à extraire.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur des colonnes filtrées selon le motif regex.
    """
    return torch.tensor(
        allData_df.filter(regex=regex_pattern).values,
        dtype=tensorDType
    )

def get_target_volume(allData_df: pd.DataFrame, tensorDType : torch.dtype | None = None) -> torch.Tensor:
    """
    Extrait le tenseur de sortie pour les volumes d'eau à prédire (target).

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        torch.Tensor: Tenseur des volumes d'eau à prédire (target).
    """
    return torch.tensor(
        allData_df.filter(regex=f'^T[0-9]+_{modified_volume_col_name}$').values,
        dtype=tensorDType
    )

def get_all_data_tensors(allData_df: pd.DataFrame, tensorDType : torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extrait tous les tenseurs d'entrée et de sortie à partir du DataFrame des séquences de données préparées.

    Args:
        allData_df (pd.DataFrame): DataFrame contenant les séquences de données préparées.
        tensorDType (torch.dtype | None): Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - Tenseur des volumes d'eau des jours précédents (lagged).
            - Tenseur des températures observées des jours précédents (lagged).
            - Tenseur des températures prévues pour les jours à venir (forecast/horizon).
            - Tenseur des volumes d'eau à prédire (target).
    """

    """
    Préparer les tenseurs d'entrée et de sortie pour le modèle.
    - input_volume_lagged_tensor: volumes d'eau des jours précédents (lagged)
    - input_temperature_lagged_tensor: températures observées des jours précédents (lagged)
    - input_temperature_forecast_tensor: températures prévues pour les jours à venir (forecast)
    - target_volume_tensor: volumes d'eau à prédire (target)

    Timeline diagram:
    Time --> T-[n_lookback_days] ... T-2, T-1 |    T0   | T1, T2, ... T[horizon-1]
            |              Passé              | Présent |          Futur          |
            |---------------------------------|---------|-------------------------|
                    X (lagged inputs)         |       X (forecast inputs)
                                              |       Y (target outputs)                                     
    """
    input_volume_lagged_tensor = get_input_volume_lagged(allData_df, tensorDType)
    input_temperature_lagged_tensor = get_input_temperature_lagged(allData_df, tensorDType)
    input_temperature_forecast_tensor = get_input_temperature_forecast(allData_df, tensorDType)
    target_volume_tensor = get_target_volume(allData_df, tensorDType)

    return input_volume_lagged_tensor, input_temperature_lagged_tensor, input_temperature_forecast_tensor, target_volume_tensor
    

def get_sequence(ACER_csv_path: str, n_lookback_days: int = 3, horizon: int = 1, padding_days: int = 5, tensorDType : torch.dtype | None = None, norm_by_dayly_deviation=False, verbose=False) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Génère une séquence de données ACER pour l'entraînement de modèles de machine learning.

    Args:
        ACER_csv_path (str): Chemin vers le fichier CSV des données ACER.
        n_lookback_days (int): Nombre de jours précédents à inclure dans chaque séquence.
        horizon (int): Nombre de jours dans le futur à prédire.
        padding_days (int): Nombre de jours de padding à ajouter au début et à la fin des séquences.
        tensorDType: Type de données des tenseurs PyTorch. Si None, utilise le type par défaut.
    Returns:
        id_date_system (DataFrame)(num_samples, 2): DataFrame des identifiants de date et de système.
        input_volume_lagged_tensor (tensor)(num_samples, n_lookback_days): Séquences des volumes d'eau des jours précédents. (T-n, ... T-2, T-1)
        input_temperature_lagged_tensor (tensor)(num_samples, n_lookback_days* 24): Séquences des temperatures observées des jours précédents (T-n, ... T-2, T-1).
        input_temperature_forecast_tensor (tensor)(num_samples, horizon*24): Séquences des temperatures prévues pour les jours à venir (T0, T1, ... T[horizon-1]).
        target_volume_tensor (tensor)(num_samples, horizon): Volumes d'eau à prédire. (Y0, Y1, ... Y[horizon-1])
    """
    # Charger les données ACER depuis le fichier CSV
    ACER_df = get_ACER_df(ACER_csv_path)

    volume_column_name = 'volume entaille'
    if norm_by_dayly_deviation:
        ACER_df, past_years_df = ACER_df_append_past_years_norm(ACER_df)
        volume_column_name = 'volume_norm'
    
    allData_df = prep_sequence(
        ACER_df=ACER_df,
        volume_column_name=volume_column_name,
        n_lookback_days=n_lookback_days,
        horizon=horizon,
        start_padding_days=padding_days,
        end_padding_days=0,
        verbose=verbose
    )

    id_date_system = get_id_date_system(allData_df, past_years_df if norm_by_dayly_deviation else None)
    input_volume_lagged_tensor, input_temperature_lagged_tensor, input_temperature_forecast_tensor, target_volume_tensor = get_all_data_tensors(allData_df, tensorDType)
    
    return id_date_system, input_volume_lagged_tensor, input_temperature_lagged_tensor, input_temperature_forecast_tensor, target_volume_tensor