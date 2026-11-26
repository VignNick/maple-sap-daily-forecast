from utils.yaml.YAML_registry import yaml_serializable
from utils.yaml.YAML_resolver import YAMLResolver as YAMLRes
import os


@yaml_serializable(builder_method=lambda target, kwargs: target.from_dict(kwargs), save_method=lambda obj: obj.to_dict())
class ParamGrid:
    '''
    Classe pour gérer la génération de grilles de paramètres pour les tests sur des modèles.
    Les utilisateurs peuvent fournir une grille de paramètres sous forme de dictionnaire;
    - Où les clés primaires sont les sous-dictionnaires contenant un ensemble de paramètres
        pour une catégorie
    - Ces sous-dictionnaires contient les paramètres et les valeurs sont des listes de valeurs
        possibles pour cette catégorie de paramètres.
    
    Chaque combinaison de paramètres est générée à partir de la grille et peut être utilisée pour
    tester différents modèles ou configurations.
     - Les combinaisons sont aussi générées pour toutes combinaisons des clés primaires.

    '''

    _ROOT_KEY = "__root__"
    
    def __init__(self, param_grid: dict):
        '''
        Initialise la classe avec une grille de paramètres.
        
        Args:
            param_grid (dict): Un dictionnaire contenant la grille de paramètres (dictionnaires imbriqués).
        '''
        self.param_grid = param_grid
        self.__param_grid_hash = None
        self.__combinations = self.__generate_combinations()

    @classmethod
    def from_dict(cls, param_grid_dict: dict) -> 'ParamGrid':
        '''
        Crée une instance de ParamGrid à partir d'un dictionnaire de grille de paramètres.
        
        Args:
            param_grid_dict (dict): Un dictionnaire contenant la grille de paramètres (dictionnaires imbriqués).
        
        Returns:
            ParamGrid: Une instance de la classe ParamGrid initialisée avec les données du dictionnaire.
        '''
        return cls(param_grid=param_grid_dict)
    
    def to_dict(self) -> dict:
        '''
        Sérialise la grille de paramètres en un dictionnaire.
        
        Returns:
            dict: Un dictionnaire représentant la grille de paramètres.
        '''
        # Assurer que la grille ne contient pas d'éléments non sérialisables (comme des callables)
        # avant de la retourner pour un yaml dump ou autre sérialisation.
        for key, value in self.param_grid.items():
            if not isinstance(value, (list, dict)):
                raise TypeError(f"Value for key '{key}' is not serializable.")

        return self.param_grid

    @classmethod
    def normalize_grid(cls, param_grid: dict) -> dict:
        '''
        Transforme une grille plate ou mixte en grille hiérarchique.

        Exemple:
        
            {
                "lr": [0.01, 0.1],
                "max_depth": [3, 5],
                "model": {
                    "depth": [3, 5]
                }
            }

            devient ->

            {
                "__root__": {
                    "lr": [0.01, 0.1],
                    "max_depth": [3, 5]
                },
                "model": {
                    "depth": [3, 5]
                }
            }
        '''

        normalized = {}
        root_params = {}

        for key, value in param_grid.items():
            if isinstance(value, dict):
                normalized[key] = value
            else:
                root_params[key] = value

        if root_params:
            normalized[cls._ROOT_KEY] = root_params

        return normalized

    @classmethod
    def denormalize_combination(cls, combination: dict) -> dict:
        '''
        Supprime la clé __root__ d'une combinaison.

        Exemple:
            
            {
                "__root__": {
                    "lr": 0.01
                    "max_depth": 5
                },
                "model": {
                    "depth": 3 
                }
            }

            devient ->

            {
                "lr": 0.01,
                "max_depth": 5,
                "model": {
                    "depth": 3 }
            }
        '''

        if cls._ROOT_KEY not in combination:
            return combination

        result = dict(combination[cls._ROOT_KEY])

        for key, value in combination.items():
            if key != cls._ROOT_KEY:
                result[key] = value

        return result

    @classmethod
    def denormalize_combinations(cls, combinations: list[dict]) -> list[dict]:
        '''
        Applique denormalize_combination à une liste.
        '''
        return [
            cls.denormalize_combination(combo)
            for combo in combinations
        ]

    def __generate_combinations(self) -> None:
        '''
        Génère toutes les combinaisons de paramètres à partir de la grille fournie.

        '''
        from itertools import product

        if self.__param_grid_hash == hash(str(self.param_grid)) and self.__combinations is not None:
            return

        # Normalisation
        grid = self.normalize_grid(self.param_grid)

        primary_keys = list(grid.keys())

        all_combinations = {}
        for pk in primary_keys:
            sub_dict = grid[pk]

            keys = list(sub_dict.keys())
            values = [sub_dict[k] for k in keys]

            all_combinations[pk] = [
                dict(zip(keys, combo))
                for combo in product(*values)
            ]

        final_combinations = [
            {
                pk: sub_combo
                for pk, sub_combo in zip(primary_keys, combo)
            }
            for combo in product(*(all_combinations[pk] for pk in primary_keys))
        ]

        self.__combinations = final_combinations
        self.__param_grid_hash = hash(str(self.param_grid))
    
    def get_combinations(self, denormalize=True) -> list[dict]:
        '''
        Retourne toutes les combinaisons de paramètres générées à partir de la grille.
        Args:
            denormalize (bool, optional): Si True, les combinaisons retournées seront dénormalisées
                (sans la clé __root__). Par défaut, True.
        Returns:
            list[dict]: Une liste de dictionnaires représentant toutes les combinaisons de paramètres.
        '''
        self.__generate_combinations()
        return self.__combinations if not denormalize else self.denormalize_combinations(self.__combinations)
    
    def get_combinations_count(self) -> int:
        '''
        Retourne le nombre total de combinaisons de paramètres générées à partir de la grille.
        Returns:
            int: Le nombre total de combinaisons de paramètres.
        '''
        self.__generate_combinations()
        return len(self.__combinations)
    
    def get_combination(self, index: int, denormalize=True) -> dict:
        '''
        Retourne une combinaison de paramètres spécifique à partir de son index.
        Args:
            index (int): L'index de la combinaison à retourner (0-based).
            denormalize (bool, optional): Si True, la combinaison retournée sera dénormalisée
                (sans la clé __root__). Par défaut, True.
        Returns:
            dict: Un dictionnaire représentant la combinaison de paramètres à l'index spécifié.
        Raises:
            IndexError: Si l'index est en dehors des limites des combinaisons disponibles.
        '''
        self.__generate_combinations()
        if index < 0 or index >= len(self.__combinations):
            raise IndexError("Index out of range for combinations.")
        
        combination = self.__combinations[index]
        return combination if not denormalize else self.denormalize_combination(combination)

@yaml_serializable()
class DirectConstructor:
    '''
    '''
    def __init__(self, constructor: callable, args: dict):
        self.constructor = constructor
        self.args = args

    def get_object(self):
        return self.constructor(**self.args)

@yaml_serializable(builder_method=lambda target, kwargs: target.from_dict(kwargs), save_method=lambda obj: obj.to_dict())
class Experiment:
    '''
    '''
    @yaml_serializable()
    class ExpArgConstructor:
        '''
        Cette classe est responsable de construire un arguments de l'expérience à partir d'un
        constructeur d'objet et d'un mapping entre les arguments de l'expérience et les clés de la combinaison de paramètres.
         - Le constructeur est une fonction ou une classe qui prend des arguments et retourne un objet ou une configuration.
         - Le mapping est un dictionnaire où les clés sont les noms des arguments du constructeur et les valeurs sont les 
            clés correspondantes dans la combinaison de paramètres.
        '''
        def __init__(self, constructor: callable, exp_args_mapping: dict):
            '''
            Args:
                constructor (callable): Une fonction ou une classe qui prend des arguments et retourne un objet ou une configuration.
                exp_args_mapping (dict): Un dictionnaire où les clés sont les noms des arguments du constructeur et les valeurs sont
                    les clés correspondantes dans la combinaison de paramètres.
            '''
            self.constructor = constructor
            self.exp_args_mapping = exp_args_mapping

        @classmethod
        def from_dict(cls, data: dict) -> 'Experiment.ExpArgConstructor':
            '''
            Crée une instance de ExpArgConstructor à partir d'un dictionnaire.
            Args:
                data (dict): Un dictionnaire contenant les clés 'constructor' et 'exp_args_mapping'.
            Returns:
                ExpArgConstructor: Une instance de la classe ExpArgConstructor initialisée avec les données du dictionnaire.
            '''
            return cls(
                constructor = data['constructor'],
                exp_args_mapping = data['exp_args_mapping']
            )
        
        def to_dict(self) -> dict:
            '''
            Sérialise le ExpArgConstructor en un dictionnaire.
            Returns:
                dict: Un dictionnaire représentant le ExpArgConstructor.
            '''
            return {
                'constructor': self.constructor,
                'exp_args_mapping': self.exp_args_mapping
            }
    
        def get_exp_arg(self, param_combination: dict):
            '''
            Args:
                param_combination (dict): Un dictionnaire représentant une combinaison de paramètres.
            Returns:
                L'objet ou la configuration construite à partir du constructeur et du mapping.
            '''
            import inspect
            def solve_mapping(dict_of_value: dict, mapping: dict):
                _args = {}
                for mapping_key, mapping_value in mapping.items():
                    if "__$&UnpackDirect" == mapping_key:
                        for sub_param_key, sub_param_value in param_combination[mapping_value].items():
                            _args[sub_param_key] = sub_param_value
                    elif mapping_key.endswith("$&noSearch"):
                        _args[mapping_key[:-10]] = mapping_value
                    elif isinstance(mapping_value, str):
                        keys = mapping_value.split(".")
                        value = dict_of_value
                        for key in keys:
                            value = value[key]
                        _args[mapping_key] = value
                    elif isinstance(mapping_value, dict):
                        _args[mapping_key] = solve_mapping(dict_of_value, mapping_value)
                    elif isinstance(mapping_value, Experiment.ExpArgConstructor):
                        _args[mapping_key] = mapping_value.get_exp_arg(param_combination)

                return _args

            args = solve_mapping(param_combination, self.exp_args_mapping)
            if inspect.isbuiltin(self.constructor):
                return self.constructor(*args.values())
            return self.constructor(**args)

    def __init__(
            self,
            param_grid: ParamGrid,
            experiment_function: callable,
            exp_args_mapping: dict[str, ExpArgConstructor],
            comment: str = '',
            folder_path: str = os.getcwd()
        ):
        '''

        Args:
            param_grid (ParamGrid): Une instance de la classe ParamGrid contenant la grille de paramètres.
            experiment_function (callable): La fonction à exécuter pour chaque combinaison de paramètres.
            exp_args_mapping (dict[str, ExpArgConstructor]): Un dictionnaire de constructeurs d'arguments d'expérience.
            comment (str, optional): Un commentaire pour décrire l'expérience. Par défaut, une chaîne vide.
            folder_path (str, optional): Le chemin vers le dossier où les résultats de l'expérience seront sauvegardés. Par défaut, le répertoire de travail actuel.
        '''
        self.param_grid = param_grid
        self.experiment_function = experiment_function
        self.exp_args_mapping = exp_args_mapping
        self.comment = comment
        self.folder_path = folder_path

    @classmethod
    def load_folder(cls, folder_path: str, yaml_file_name: str = "experiment.yaml"):
        '''
        Charge une instance de Experiment à partir d'un dossier contenant un fichier YAML.
        Args:
            folder_path (str): Le chemin vers le dossier contenant le fichier YAML de l'expérience.
            yaml_file_name (str, optional): Le nom du fichier YAML à charger. Par défaut, "experiment.yaml".
        Returns:
            Experiment: Une instance de la classe Experiment initialisée avec les données du fichier YAML.
        '''
        yaml_file_path = os.path.join(folder_path, yaml_file_name)
        obj = cls.load_from_yaml(yaml_file_path)
        obj.folder_path = folder_path
        return obj
    
    @classmethod
    def load_from_yaml(cls, yaml_file: str):
        '''
        Crée une instance de Experiment à partir d'un fichier YAML.
        Le fichier YAML doit contenir une structure avec une clé 'param_grid' pour la grille de
        paramètres et une clé 'comment' pour le commentaire descriptif.
        Args:
            yaml_file (str): Le chemin vers le fichier YAML contenant la grille de paramètres et le
                commentaire.
        Returns:
            ParamGrid: Une instance de la classe ParamGrid initialisée avec les données du fichier YAML.
        '''
        data = YAMLRes.safe_load_yaml(yaml_file)

        return cls(
            param_grid=data.get('param_grid', {}),
            experiment_function=data.get('experiment_function', None),
            exp_args_mapping=data.get('exp_args_mapping', None),
            comment=data.get('comment', ''),
            folder_path=os.path.dirname(yaml_file)
            )
    
    def save_to_yaml(self, yaml_file: str | None = None):
        '''
        Sauvegarde la grille de paramètres et le commentaire dans un fichier YAML.
        Args:
            yaml_file (str | None): Le chemin vers le fichier YAML où les données seront sauvegardées.
                Si le fichier n'existe pas, il sera créé. Si le fichier existe déjà, il sera écrasé.
                Si None, le fichier sera sauvegardé dans le dossier de l'expérience sous le nom "experiment.yaml".
                Par défaut, None.
        '''
        data = {
            'comment': self.comment,
            'param_grid': self.param_grid,
            'experiment_function': self.experiment_function,
            'exp_args_mapping': self.exp_args_mapping,
        }

        if yaml_file is None:
            yaml_file = os.path.join(self.folder_path, "experiment.yaml")

        YAMLRes.safe_save_yaml(data, yaml_file)

    def get_combinations_count(self) -> int:
        '''
        Retourne le nombre total de combinaisons de paramètres générées à partir de la grille.
        Returns:
            int: Le nombre total de combinaisons de paramètres.
        '''
        return self.param_grid.get_combinations_count()

    def run(self, index: int):
        '''
        Exécute l'expérience pour une combinaison de paramètres spécifique.
        Args:
            index (int): L'index de la combinaison de paramètres à utiliser.
        Returns:
            Le résultat de la fonction d'expérience.
        '''
        combination = self.param_grid.get_combination(index=index)
        combination["__$&FolderPath"] = self.folder_path
        combination["__$&RunIndex"] = index
        args = {
            exp_arg: arg_constructor.get_exp_arg((combination))
            for exp_arg, arg_constructor in self.exp_args_mapping.items()
        }
        return self.experiment_function(**args)