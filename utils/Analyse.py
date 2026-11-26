import os, yaml
import pandas as pd

def get_Results(file_path: str, prefix: str | None = None) -> dict:
    with open(file_path, 'r') as f:
        results = yaml.safe_load(f)

    # Ajouter un préfixe aux clés du dictionnaire si un préfixe est fourni
    if prefix is not None:
        results = {f"{prefix}_{key}": value for key, value in results.items()}

    return results

def search_Results(directory: str, file_end: str = "results.yaml") -> list:
    results_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_end):
                results_files.append(os.path.join(root, file))
    return results_files

def aggregate_results(directory: str) -> dict:
    aggregated_results = {}
    results_files = search_Results(directory)

    for file_path in results_files:
        # Preffix le chemin relatif du fichier comme préfixe pour éviter les collisions de clés
        prefix = os.path.relpath(os.path.dirname(file_path), directory).replace(os.sep, '_')
        results = get_Results(file_path, prefix)
        aggregated_results.update(results)

        # print(f"Head of results from {file_path}: {list(results.items())[:5]}")  # Affiche les 5 premières paires clé-valeur

        # for key, value in results.items():
            # aggregated_results[key] = value

    return aggregated_results

def filter_lookback_horizon(aggregated_results: dict, lookback: list[int]|None = None, horizon: list[int]|None = None) -> dict:
    filtered_results = {}
    for key, value in aggregated_results.items():
        if lookback is not None and value["lookback"] not in lookback:
            continue
        if horizon is not None and value["horizon"] not in horizon:
            continue
        filtered_results[key] = value
    return filtered_results

def get_top_experiments(aggregated_results: dict, sort_key: str, top_n: int = 20) -> list:
    '''
    Retourne les n meilleures expériences triées par une clé spécifique.

    Args:
        aggregated_results (dict): Dictionnaire contenant les résultats agrégés de toutes les expériences
        sort_key (str): Clé utilisée pour trier les expériences (ex: "best_val_loss_orig") dans "stats" -> "best_val"
        top_n (int): Nombre d'expériences à retourner
    Returns:
        list: Liste des top_n expériences triées par la clé spécifiée
    '''
    if not aggregated_results:
        return []

    # Trier les expériences par la clé spécifiée et retourner les top_n meilleures
    if sort_key in next(iter(aggregated_results.values())):
        top_experiments = sorted(aggregated_results.items(), key=lambda x: x[1][sort_key])[:top_n]
    elif 'stats' in next(iter(aggregated_results.values())) and 'best_val' in next(iter(aggregated_results.values()))['stats']:
        top_experiments = sorted(aggregated_results.items(), key=lambda x: x[1]["stats"]["best_val"][sort_key])[:top_n]
    else:
        top_experiments = []
    return top_experiments

def to_dataframe(top_experiments: list) -> pd.DataFrame:
    '''
    Convertit une liste d'expériences en un DataFrame pandas pour une analyse plus facile.

    Args:
        top_experiments (list): Liste des expériences à convertir, où chaque élément est un tuple (clé, valeur)
    Returns:
        pd.DataFrame: DataFrame contenant les informations des expériences
    '''

    if not top_experiments:
        return pd.DataFrame()  # Retourne un DataFrame vide si la liste est vide
    if 'stats' in top_experiments[0][1]:
        def get_value(exp, key):
            return exp[1]["stats"]["best_val"].get(key, None)
    else:
        def get_value(exp, key):
            return exp[1].get(key, None)


    df_top_experiments = pd.DataFrame({
        "Experiment": [k for k, v in top_experiments],
        "Best Val Loss": [get_value(exp, "best_val_loss") for exp in top_experiments],
        "Best Val Loss Orig": [get_value(exp, "best_val_loss_orig") for exp in top_experiments],
        "Best Val Epoch": [v["best_val_epoch"] for k, v in top_experiments],
        "Best Val WAPE": [get_value(exp, "best_val_wape") for exp in top_experiments],
        "Configuration": [v["configuration"] for k, v in top_experiments],
        # "Mixing Hidden Dim": [v["mixing_hidden_dim"] for k, v in top_experiments],
        # "Learning Rate": [v["learning_rate"] for k, v in top_experiments]
    })
    return df_top_experiments

def get_top_experiments_dataframe(directory: str, sort_key: str, top_n: int = 20, lookback: list[int]|None = None, horizon: list[int]|None = None) -> pd.DataFrame:
    '''
    Retourne un DataFrame contenant les n meilleures expériences triées par une clé spécifique.

    Args:
        directory (str): Chemin vers le répertoire contenant les résultats des expériences
        sort_key (str): Clé utilisée pour trier les expériences (ex: "best_val_loss_orig") dans "stats" -> "best_val"
        top_n (int): Nombre d'expériences à retourner
        lookback (list[int]|None): Liste des valeurs de lookback à inclure
        horizon (list[int]|None): Liste des valeurs de horizon à inclure
    Returns:
        pd.DataFrame: DataFrame contenant les informations des top_n expériences triées par la clé spécifiée
    '''
    return to_dataframe(
            get_top_experiments(
                filter_lookback_horizon(
                    aggregate_results(directory),
                    lookback,
                    horizon
                ),
                sort_key,
                top_n
            )
        )