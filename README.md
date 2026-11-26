# MapleAI

Projet de prévision des volumes journaliers de sève d'érable à partir des
observations ACER et, selon le modèle, de variables météorologiques. Le dépôt
contient les outils de préparation des données, des modèles PyTorch, le code
d'entraînement et des notebooks d'expérimentation et d'analyse.

> **Données non incluses :** les fichiers ACER, les bases météo, les modèles
> entraînés et les résultats d'expérience ne sont pas distribués dans ce dépôt.
> Il faut disposer des données nécessaires avant d'exécuter les exemples.

## Contenu du dépôt

```text
classes/
  ACER_csv.py           Conversion de classeurs ACER en CSV
  ACER_helper.py        Chargement, préparation et séquençage des données
  ACER_data_handler.py  Découpage train/validation et datasets PyTorch
  maple_helper.py       Normalisation des tenseurs
  meteo_db.py           Cache SQLite et récupération Open-Meteo
models/
  acer_conf_lstm.py     Modèle LSTM pour les données ACER
  TSMixer.py            Composants TSMixer et variantes
  CustomMixer.py        Modèles combinant météo et données ACER
  MeteoProcess.py       Encodage des séries météorologiques
  training.py           Entraînement et métriques de régression
  ParamGrid.py          Expériences et recherche de paramètres
utils/
  Analyse.py            Agrégation et classement des résultats
  SapGraph.py           Graphiques des prédictions
  yaml/                 Sérialisation des modèles et expériences
TrainingExp.ipynb       Configurations d'expériences d'entraînement
_AnalyseData.ipynb      Exploration et analyse des données
_AnalyseResultat.ipynb  Comparaison des expériences et graphiques
```

## Prérequis

- Python 3.10 ou plus récent.
- PyTorch. Choisissez la version correspondant à votre système et à votre
  matériel (CPU, CUDA ou autre accélérateur) dans les
  [instructions officielles d'installation de PyTorch](https://pytorch.org/get-started/locally/).
- Les dépendances Python des modules et notebooks.

Depuis la racine du dépôt, créez un environnement virtuel et installez les
dépendances :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Installer PyTorch selon la configuration de la machine (requirements-amd.txt -> dépendences PyTorch pour AMD).
python -m pip install -r requirements-amd.txt
```

## Préparer les données ACER

Le convertisseur `classes/ACER_csv.py` lit les feuilles d'un classeur Excel et
produit un CSV nettoyé. Les noms de feuilles doivent représenter les années;
le script demande confirmation si l'année des dates ne correspond pas à celle
de la feuille. Pour lancer le convertisseur de façon interactive :

```bash
python -m classes.ACER_csv
```

Le fichier CSV obtenu contient notamment les colonnes `date`, `system` et
`volume entaille` utilisées par le pipeline ACER. Le gestionnaire de données
recherche par défaut le fichier suivant :

```text
Data/ACER_2000-2024.csv
```

Placez-y un CSV compatible avec ces colonnes. Si vous avez utilisé le nom de
sortie par défaut du convertisseur, vous pouvez par exemple faire :

Le nom `ACER_2000-2024.csv` est un chemin par défaut, pas un fichier fourni par
le dépôt : adaptez `ACER_csv_path` dans votre code si votre fichier se trouve
ailleurs.

## Préparer et explorer un jeu de données

`ACERDataModule` construit les séquences à partir des observations, sépare les
échantillons d'entraînement et de validation, puis calcule les statistiques de
normalisation à partir de l'ensemble d'entraînement. Exemple minimal :

```python
from classes.ACER_data_handler import ACERDataModule

data_module = ACERDataModule(
    ACER_csv_path="./Data/ACER_2000-2024.csv",
    n_lookback_days=3,
    horizon=1,
    val_split=0.2,
)
data_module.get_ready()
train_dataset, validation_dataset = data_module.get_train_val_datasets()

print(f"Échantillons : {data_module.length}")
print(f"Entraînement : {len(train_dataset)}")
print(f"Validation : {len(validation_dataset)}")
```

`n_lookback_days` contrôle la fenêtre historique et `horizon` le nombre de
jours à prédire. `val_split` accepte une fraction (par exemple `0.2`) ou une
liste d'années à réserver pour la validation. D'autres options permettent
notamment de choisir l'heure d'échantillonnage, l'augmentation des données et
la normalisation des volumes ou des températures.

## Entraîner et analyser

Le notebook [TrainingExp.ipynb](./TrainingExp.ipynb) contient des configurations
pour les modèles LSTM et TSMixer, ainsi que des expériences utilisant des
variables météo. Ouvrez-le depuis la racine du projet et exécutez les cellules
dans l'ordre. Les configurations d'expérience permettent de définir les
paramètres du modèle, du jeu de données et de l'entraînement; les résultats et
modèles sauvegardés sont écrits dans des dossiers comme `models_testing/`.

Les autres notebooks servent à explorer le jeu de données et à comparer les
expériences :

- [_AnalyseData.ipynb](./_AnalyseData.ipynb) : statistiques, courbes et
  comportement des volumes.
- [_AnalyseResultat.ipynb](./_AnalyseResultat.ipynb) : classement des résultats
  et visualisation des prédictions.

Les dossiers d'expériences et de résultats ne sont pas inclus dans le dépôt.
Certains chemins d'expérience inscrits dans les notebooks sont propres aux
exécutions antérieures et peuvent nécessiter une adaptation.

## Données météorologiques

Les modèles météo s'appuient sur `MeteoDB`, qui conserve les observations dans
une base SQLite (par défaut `Data/meteo_dbs.db`) et récupère les heures absentes
depuis l'API d'archive Open-Meteo. Ces modèles nécessitent une connexion réseau
lorsque des observations doivent être téléchargées. La base et le cache météo
sont générés localement; ils ne sont pas inclus dans le dépôt.