import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from classes.ACER_data_handler import ACERDataModule, ACERDataset
import yaml
from .model import BaseModel
from .ParamGrid import Experiment
import os

# ------------------------------------------------------
# Metriques de régression
# ------------------------------------------------------
class RegressionMetrics:
    def __init__(self, device):
        self.device = device

        self.sse = torch.zeros(1, device=device)
        self.sae = torch.zeros(1, device=device)
        self.sum_y = torch.zeros(1, device=device)
        self.sum_y_squared = torch.zeros(1, device=device)
        self.n = torch.zeros(1, device=device)

    def update(self, y_pred, y_true):
        diff = y_pred - y_true

        self.sse += torch.sum(diff ** 2)
        self.sae += torch.sum(torch.abs(diff))
        self.sum_y += torch.sum(y_true)
        self.sum_y_squared += torch.sum(y_true ** 2)
        self.n += y_true.numel()

    def compute(self):
        eps = 1e-12
        n = torch.clamp(self.n, min=1)

        mse = self.sse / n
        rmse = torch.sqrt(mse)

        # WAPE (Weighted Absolute Percentage Error)
        wape = self.sae / (torch.abs(self.sum_y) + eps)

        # RMSE% (Root Mean Square Error Percentage)
        mean_y = self.sum_y / n
        rmse_percent = rmse / (torch.abs(mean_y) + eps)

        # SST (Total Sum of Squares)
        sst = self.sum_y_squared - (self.sum_y ** 2) / n

        # R2 (Coefficient of Determination)
        r2 = 1 - self.sse / (sst + eps)

        # Move to CPU only once
        return {
            "MSE": mse.item(),
            "RMSE": rmse.item(),
            "RMSE%": (100 * rmse_percent).item(),
            "WAPE": (100 * wape).item(),
            "R2": r2.item(),
        }

# ------------------------------------------------------
# Helper pour construire les param groups pour AdamW
# ------------------------------------------------------
def make_optimizer_parameters(
    model: nn.Module,
    *,
    base_lr: float = 1e-3,
    lstm_lr: float | None = None,
    weight_decay: float = 1e-2,
) -> list[dict]:
    """Crée les groupes de paramètres pour AdamW avec no decay pour biases / LayerNorm."""
    param_groups = []
    handled_params = set()

    def is_norm_parameter(module_name: str, param_name: str) -> bool:
        return "norm" in module_name.lower() or param_name.endswith(".bias")

    # LSTM first
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            decay, no_decay = [], []
            for pname, p in module.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                if is_norm_parameter(module_name, pname):
                    no_decay.append(p)
                else:
                    decay.append(p)
                handled_params.add(id(p))
            if decay:
                param_groups.append({
                    "params": decay,
                    "lr": lstm_lr if lstm_lr is not None else base_lr,
                    "weight_decay": weight_decay
                })
            if no_decay:
                param_groups.append({
                    "params": no_decay,
                    "lr": lstm_lr if lstm_lr is not None else base_lr,
                    "weight_decay": 0.0
                })

    # All other modules
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LSTM):
            continue
        decay, no_decay = [], []
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in handled_params:
                continue
            if is_norm_parameter(module_name, pname):
                no_decay.append(p)
            else:
                decay.append(p)
            handled_params.add(id(p))
        if decay:
            param_groups.append({"params": decay, "lr": base_lr, "weight_decay": weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": base_lr, "weight_decay": 0.0})

    return param_groups

# ------------------------------------------------------
# Training ARGS dataclass
# ------------------------------------------------------
class TrainingArgs:
    def __init__(
        self,
        num_epochs: int = 50,                           # nombre d'époques
        batch_size: int = 32,                           # taille des batchs
        learning_rate: float = 1e-3,                    # taux d'apprentissage
        lstm_lr: float | None = None,                   # taux d'apprentissage spécifique pour LSTM (si None, utilise learning_rate)
        weight_decay: float = 1e-2,                     # weight decay pour l'optimiseur
        criterion: nn.Module | None = None,             # Fonction de perte; par défaut MSELoss
        device: torch.device | None = None,             # 'cuda' ou 'cpu'; par défaut auto
        dataType: torch.dtype | None = torch.float32,   # Type de données (float32, float16, bfloat16); par défaut float32
        save_path: str | None = None,                   # Chemin pour sauvegarder le modèle entraîné; si None, ne pas sauvegarder
        verbose: bool = True,                           # Afficher la progression
        best_val_loss: float = float("inf"),            # Meilleure perte de validation à battre
        early_stopping_patience: int = 100,             # Number of epochs without improvement
        early_stopping_min_delta: float = 1e-5,         # Minimum improvement to count as progress
        early_stopping_min_epoch: int = 0               # Minimum d'époques avant de commencer à vérifier l'early stopping
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lstm_lr = lstm_lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.device = device
        self.dataType = dataType
        self.save_path = save_path
        self.verbose = verbose
        self.best_val_loss = best_val_loss
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_min_epoch = early_stopping_min_epoch

    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)
    
    def to_safe_dict(self):
        # Convertir les types non sérialisables en YAML en des formats lisibles
        dict_copy = self.to_dict().copy()
        if isinstance(dict_copy['device'], torch.device):
            dict_copy['device'] = str(dict_copy['device'])
        if isinstance(dict_copy['criterion'], nn.Module):
            dict_copy['criterion'] = dict_copy['criterion'].__class__.__name__
        if isinstance(dict_copy['dataType'], torch.dtype):
            dict_copy['dataType'] = str(dict_copy['dataType'])
        return dict_copy
    
    @classmethod
    def from_safe_dict(cls, safe_dict: dict):
        config = safe_dict.copy()

        # Restore device
        if isinstance(config.get("device"), str):
            config["device"] = torch.device(config["device"])

        # Restore dtype
        if isinstance(config.get("dataType"), str):
            dtype_str = config["dataType"].replace("torch.", "")
            if hasattr(torch, dtype_str):
                config["dataType"] = getattr(torch, dtype_str)

        # Restore criterion
        if isinstance(config.get("criterion"), str):
            criterion_name = config["criterion"]
            if hasattr(nn, criterion_name):
                config["criterion"] = getattr(nn, criterion_name)()
            else:
                raise ValueError(f"Unknown criterion: {criterion_name}")

        return cls(**config)

# ------------------------------------------------------
# Training function
# ------------------------------------------------------
def train_acer_data_module(
    model: BaseModel,                                       # Modèle PyTorch (ACER_LSTM ou ACER_LSTM_WithForecast)
    data_module: ACERDataModule,                            # DataModule contenant les datasets d'entraînement et de validation
    training_args: TrainingArgs,                            # Arguments d'entraînement
    ):
    """
    Sequence d'entraînement d'un modèle pour les données généré par ACERDataModule.

    Args:
        model (BaseModel)                   : Modèle PyTorch (ACER_LSTM ou ACER_LSTM_WithForecast)
        data_module (ACERDataModule)        : DataModule contenant les datasets d'entraînement et de validation
        training_args (TrainingArgs)        : Arguments d'entraînement encapsulés dans une dataclass

    Returns:
        modèle entraîné
        historique d'entraînement (pertes)
    """
    data_module.get_ready()
    train_dataset, validation_dataset = data_module.get_train_val_datasets()
    # --------- Vérification de cohérence ----------
    # Vérifier si les datasets sont issus du même data module
    assert train_dataset._dm == validation_dataset._dm, "Les datasets d'entraînement et de validation doivent provenir du même data module."
    dm = train_dataset._dm
    
    # Est-ce que la normalisation des volumes basée sur la déviation journalière des années précédentes est utilisée
    norm_by_dayly_deviation = dm.norm_by_dayly_deviation

    # trouver la position du Tenseur target_volume
    target_idx = None
    for i, field in enumerate(dm.return_fields):
        if field.startswith('target_volume'):
            target_idx = i
            break
    # target_idx = dm.return_fields.index('target_volume')
    volume_journalier_moyen_idx = None
    if 'volume_journalier_moyen' in dm.return_fields:
        volume_journalier_moyen_idx = dm.return_fields.index('volume_journalier_moyen')
        # Vérifier que l'index est le dernier
        assert volume_journalier_moyen_idx == len(dm.return_fields) - 1, "Le tenseur 'volume_journalier_moyen' doit être le dernier dans les datasets."
        assert target_idx == len(dm.return_fields) - 2, "Le tenseur 'target_volume' doit être l'avant-dernier dans les datasets."
    else:
        assert target_idx == len(dm.return_fields) - 1, "Le tenseur 'target_volume' doit être le dernier dans les datasets."
    
    # --------- Préparation device / type ----------
    if training_args.device is None:
        training_args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(training_args.device, training_args.dataType)

    train_dataset.ensure_precached(training_args.device, training_args.dataType)
    validation_dataset.ensure_precached(training_args.device, training_args.dataType)

    train_loader = DataLoader(train_dataset, batch_size=training_args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation_dataset, batch_size=training_args.batch_size, shuffle=False, num_workers=0)

    # --------- Perte et optimiseur ----------
    if training_args.criterion is None:
        training_args.criterion = nn.MSELoss()
    optimizer_params = make_optimizer_parameters(model, base_lr=training_args.learning_rate, lstm_lr=training_args.lstm_lr, weight_decay=training_args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_params)

    # --------- Préparation historique ----------
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse_orig': [],
        'val_mse_orig': [],
        'train_rmse_orig': [],
        'val_rmse_orig': [],
        'train_wape': [],
        'val_wape': [],
        'train_rmse_pct': [],
        'val_rmse_pct': [],
        'train_r2': [],
        'val_r2': [],
        }
    best_val_loss = float("inf")
    best_val_epoch = -1
    epochs_without_improvement = 0
    checkpoint = {
        'torch_version': str(torch.__version__),
        'model_class': model.__class__.__name__,
        'model_config': model.to_dict(),
        'data_module_config': data_module.to_dict(),
        'training_args': training_args.to_safe_dict(),
        'optimizer_class': "AdamW avec parametres personnalises",
    }

    criterion = training_args.criterion
    save_path = training_args.save_path
    if save_path is not None:
        save_path, ext = os.path.splitext(training_args.save_path)

    for epoch in range(0, training_args.num_epochs):
        # -------- Entraînement --------
        model.train()
        train_loss = 0.0
        # train_loss_orig = 0.0
        train_metrics = RegressionMetrics(device=training_args.device)
        for batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(*batch[:target_idx])
            loss = criterion(y_pred, batch[target_idx])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch[0].size(0)

            # Calcul de la perte sur l'échelle originale
            y_pred_orig = train_dataset.inverse_normalize_volume(y_pred)
            y_batch_orig = train_dataset.inverse_normalize_volume(batch[target_idx])
            if norm_by_dayly_deviation:
                factor = batch[volume_journalier_moyen_idx].unsqueeze(1)
                y_pred_orig = y_pred_orig * factor
                y_batch_orig = y_batch_orig * factor
            # train_loss_orig += ((y_pred_orig - y_batch_orig) ** 2).mean().item() * batch[0].size(0)
            train_metrics.update(y_pred_orig, y_batch_orig)

        train_loss /= len(train_dataset)
        # train_loss_orig /= len(train_dataset)
        train_results = train_metrics.compute()

        # -------- Validation --------
        model.eval()
        val_loss =0.0
        # val_loss_orig = 0.0
        with torch.no_grad():
            val_metrics = RegressionMetrics(device=training_args.device)
            for batch in val_loader:
                y_pred = model(*batch[:target_idx])
                loss = criterion(y_pred, batch[target_idx])
                val_loss += loss.item() * batch[0].size(0)

                # Calcul de la perte sur l'échelle originale
                y_pred_orig = validation_dataset.inverse_normalize_volume(y_pred)
                y_batch_orig = validation_dataset.inverse_normalize_volume(batch[target_idx])
                if norm_by_dayly_deviation:
                    factor = batch[volume_journalier_moyen_idx].unsqueeze(1)
                    y_pred_orig = y_pred_orig * factor
                    y_batch_orig = y_batch_orig * factor
                # val_loss_orig += ((y_pred_orig - y_batch_orig) ** 2).mean().item() * batch[0].size(0)
                val_metrics.update(y_pred_orig, y_batch_orig)
                
        val_loss /= len(validation_dataset)
        # val_loss_orig /= len(validation_dataset)
        val_results = val_metrics.compute()

        # -------- Enregistrement de l'historique --------
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse_orig'].append(train_results['MSE'])
        history['val_mse_orig'].append(val_results['MSE'])
        history['train_rmse_orig'].append(train_results['RMSE'])
        history['val_rmse_orig'].append(val_results['RMSE'])
        history['train_wape'].append(train_results['WAPE'])
        history['val_wape'].append(val_results['WAPE'])
        history['train_rmse_pct'].append(train_results['RMSE%'])
        history['val_rmse_pct'].append(val_results['RMSE%'])
        history['train_r2'].append(train_results['R2'])
        history['val_r2'].append(val_results['R2'])

        # -------- Sauvegarde du meilleur modèle --------
        improvement = best_val_loss - val_loss
        if improvement > training_args.early_stopping_min_delta:
            best_val_loss = val_loss
            best_val_epoch = epoch
            epochs_without_improvement = 0

            if save_path is not None and best_val_loss < training_args.best_val_loss:
                checkpoint['model_state_dict'] = model.state_dict()
                checkpoint['best_val_loss'] = best_val_loss
                checkpoint['best_val_epoch'] = best_val_epoch
                checkpoint['stats'] = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mse_orig': train_results['MSE'],
                    'val_mse_orig': val_results['MSE'],
                    'train_rmse_orig': train_results['RMSE'],
                    'val_rmse_orig': val_results['RMSE'],
                    'train_wape': train_results['WAPE'],
                    'val_wape': val_results['WAPE'],
                    'train_rmse_pct': train_results['RMSE%'],
                    'val_rmse_pct': val_results['RMSE%'],
                    'train_r2': train_results['R2'],
                    'val_r2': val_results['R2'],
                }
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(checkpoint, save_path+'.pt')

                # Also save a copy as a yaml file for easier reading (without the state dict)
                checkpoint_yaml = {k: v for k, v in checkpoint.items() if k != 'model_state_dict' and k != 'optimizer_state_dict'}
                with open(save_path+'.yaml', 'w') as f:
                    yaml.dump(checkpoint_yaml, f)
        elif epoch >= training_args.early_stopping_min_epoch:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_args.early_stopping_patience:
                if training_args.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}. "
                          f"Best epoch was {best_val_epoch} with val_loss={best_val_loss:.6f}")
                break

        if training_args.verbose and epoch % 5 == 0:
            print(
                f"Epoch {epoch}/{training_args.num_epochs} : "
                f"\t Train Loss: {train_loss:.6f}\t- Train Loss Orig: {train_results['MSE']:.6f}"
                f"\t Val Loss:   {val_loss:.6f}\t- Val Loss Orig:   {val_results['MSE']:.6f}\t- Val WAPE: {val_results['WAPE']:.2f}%\t- Val RMSE%: {val_results['RMSE%']:.2f}%"
                )

    return model, history, best_val_loss, best_val_epoch, epoch

class Exp_Args:
    def __init__(
        self,
        path: str,
        model_base_folder: str | None = None,
        model_folder_division: list[str] | None = None,
        nb_trials: int = 3,
        exp_number: int = 1,
    ):
        self.path = path
        self.model_save_folder = model_base_folder
        self.nb_trials = nb_trials
        self.exp_number = exp_number
        

        if self.model_save_folder is None:
            self.model_save_folder = "__$&ExpPath/saved_models/"
        if self.model_save_folder.startswith("__$&ExpPath/"):
            self.model_save_folder = os.path.join(self.path, self.model_save_folder[12:])
        if model_folder_division is not None:
            for division, value in model_folder_division.items():
                if value is not None:
                    self.model_save_folder = os.path.join(self.model_save_folder, f"{division}_{value}")

def exp_acer_data_module(
    model_ExpConstructor: Experiment.ExpArgConstructor,
    data_module: ACERDataModule,
    training_args: TrainingArgs,
    exp_args: Exp_Args,
    ):
    import textwrap

    result_path = os.path.join(exp_args.path, "results.yaml")
    training_args.save_path = os.path.join(exp_args.model_save_folder, f"experiment_{exp_args.exp_number}")

    os.makedirs(exp_args.model_save_folder, exist_ok=True)

    best_val_loss = float("inf")
    best_output = None
    best_run = None

    print(f"=== Starting experiment {exp_args.exp_number} ===")
    # Tester nb_try fois pour faire varier l'initialisation du modèle et la répartition aléatoire des données
    
    for run in range(exp_args.nb_trials):
        print(f"\tRun {run+1}/{exp_args.nb_trials} >>> ", end="")
        _model = model_ExpConstructor.get_object()
        
        _exp_out = dict(
            zip(
                ("trained_model", "history", "best_val_loss", "best_val_epoch", "last_epoch"),
                train_acer_data_module(
                    model=_model,
                    data_module=data_module,
                    training_args=training_args,
                )
            )
        )
        print(
            f"\tVal loss: {_exp_out['best_val_loss']:.6f} & Val loss orig: {_exp_out['history']['val_mse_orig'][_exp_out['best_val_epoch']]:.6f}"
            f" at epoch {_exp_out['best_val_epoch']} out of {_exp_out['last_epoch']}.")
        
        # Verifier si c'est la meilleure exécution jusqu'à présent
        if _exp_out['best_val_loss'] < best_val_loss:
            best_val_loss = _exp_out['best_val_loss']
            training_args.best_val_loss = best_val_loss  # Mettre à jour la meilleure perte de validation pour les runs suivants
            best_output = _exp_out
            best_run = run + 1

    # Sauvegarder le meilleur historique pour cette configuration d'expérience
    with open(training_args.save_path + "history.yaml", 'w') as f:
        yaml.dump(best_output["history"], f, default_flow_style=False, sort_keys=False, width=1000)
        

    # Sauvegarder les résultats de la meilleure exécution pour cette configuration dans un fichier YAML
    with open(result_path, "a") as f:
        to_save = {
            f"experiment_{exp_args.exp_number}": {
                "horizon": data_module.horizon,
                "best_val_epoch": best_output["best_val_epoch"],
                "stats": {
                    "best_val": {
                        "best_val_loss": best_output["best_val_loss"],
                        "best_val_loss_orig": best_output["history"]["val_mse_orig"][best_output["best_val_epoch"]],
                        "best_val_wape": best_output["history"]["val_wape"][best_output["best_val_epoch"]],
                        "best_val_rmse": best_output["history"]["val_rmse_orig"][best_output["best_val_epoch"]],
                        "best_val_rmse_pct": best_output["history"]["val_rmse_pct"][best_output["best_val_epoch"]],
                    },
                    "best_train": {
                        "best_train_loss": best_output["history"]["train_loss"][best_output["best_val_epoch"]],
                        "best_train_loss_orig": best_output["history"]["train_mse_orig"][best_output["best_val_epoch"]],
                        "best_train_wape": best_output["history"]["train_wape"][best_output["best_val_epoch"]],
                        "best_train_rmse": best_output["history"]["train_rmse_orig"][best_output["best_val_epoch"]],
                        "best_train_rmse_pct": best_output["history"]["train_rmse_pct"][best_output["best_val_epoch"]],
                    }
                },
                "save_path": training_args.save_path,
                "configuration": textwrap.dedent(f"""
                    horizon: {data_module.horizon}, 
                    lookback_days: {data_module.n_lookback_days}, 
                    norm_by_dayly_deviation: {data_module.norm_by_dayly_deviation},
                    normalize_volume: {data_module.normalize_volume},
                    normalize_temperature: {data_module.normalize_temperature},
                    dtype: {training_args.dataType},
                """).replace('\n', ''),
                "best_run": best_run,
                "epoch_runned": best_output["last_epoch"],
            }
        }
        f.write(yaml.dump(to_save, default_flow_style=False, sort_keys=False, width=1000))

    print(
        f"\tExperiment {exp_args.exp_number} completed.\n\t"
        f"\tBest val loss: {best_output['best_val_loss']:.6f} at epoch {best_output['best_val_epoch']} on run {best_run}."
        f"\tVal Loss:   {best_output['history']['val_loss'][best_output['best_val_epoch']]:.6f}"
        f"\t- Val Loss Orig:   {best_output['history']['val_mse_orig'][best_output['best_val_epoch']]:.6f}"
        f"\t- Val WAPE: {best_output['history']['val_wape'][best_output['best_val_epoch']]:.2f}%"
        f"\t- Val RMSE%: {best_output['history']['val_rmse_pct'][best_output['best_val_epoch']]:.2f}%"
    )

    best_output["trained_model_path"] = training_args.save_path + ".pt"

    return best_output