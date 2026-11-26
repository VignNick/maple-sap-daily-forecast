from matplotlib import dates
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from models.training import RegressionMetrics
import numpy as np

from classes.ACER_data_handler import ACERDataset
from models.model import BaseModel

def sap_horizon_1_graph(model: BaseModel, dataset: ACERDataset, figsize=(12, 6), showDates=10, datesAngle=45):
    _dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    dm = dataset._dm

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

    real_values = []
    predicted_values = []

    model.eval()
    with torch.no_grad():
        _metrics_track = RegressionMetrics(device=dataset._cached_device)
        for batch in _dataloader:
            y_pred = model(*batch[:target_idx])  # Prédiction du modèle

            y_pred_orig = dataset.inverse_normalize_volume(y_pred)
            y_batch_orig = dataset.inverse_normalize_volume(batch[target_idx])

            if norm_by_dayly_deviation:
                factor = batch[volume_journalier_moyen_idx].unsqueeze(1)
                y_pred_orig = y_pred_orig * factor
                y_batch_orig = y_batch_orig * factor

            _metrics_track.update(y_pred_orig, y_batch_orig)

            predicted_values.append(y_pred_orig.cpu()[:,:,0].detach().to(torch.float32).numpy()[0,0])
            real_values.append(y_batch_orig.cpu()[:,:,0].detach().to(torch.float32).numpy()[0,0])

        _metrics = _metrics_track.compute()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(real_values, label="Real Values", marker="o")
    ax.plot(predicted_values, label="Predicted Values", marker="x")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Volume (litres)")
    ax.set_title("Real vs Predicted Values")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),  # same x, higher y
        borderaxespad=0
    )

    if showDates > 0:
        id_date_sys = dataset._resolve_field("id_date_system")
        dates = id_date_sys[:, 0]
        ax.set_xticks(range(0, len(dates), showDates))
        ax.set_xticklabels(
            [str(d)[:10] for d in dates[::showDates]],
            rotation=datesAngle,
            ha="right",
        )

    if True:  # Afficher les métriques sur le graphique
        metrics_text = (
            "        Metrics\n"
            "-------------------------\n"
            f"MSE:        {_metrics['MSE']:.4f}\n"
            f"RMSE:      {_metrics['RMSE']:.4f}\n"
            f"WAPE:      {_metrics['WAPE']:.2f}%\n"
            f"RMSE%:   {_metrics['RMSE%']:.2f}%"
        )
        # Add metrics in top-left corner
        ax.text(
            1.05, 0.50,          # x=2%, y=98% of axes
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="lightgray",
                edgecolor="steelblue",
                alpha=0.5
            )
        )

    plt.close(fig)  # Fermer la figure pour éviter l'affichage dans les notebooks

    return fig, ax

def __get_DM_info(dataset: ACERDataset):
    dm = dataset._dm

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

    return norm_by_dayly_deviation, target_idx, volume_journalier_moyen_idx

def __get_prediction_and_real_values(model: BaseModel, dataset: ACERDataset):
    _dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    norm_by_dayly_deviation, target_idx, volume_journalier_moyen_idx = __get_DM_info(dataset)

    real_values = []
    predicted_values = []
    dataloader_length = len(_dataloader)

    model.eval()
    with torch.no_grad():
        _metrics_track = RegressionMetrics(device=dataset._cached_device)
        for i, batch in enumerate(_dataloader):
            y_pred = model(*batch[:target_idx])  # Prédiction du modèle

            y_pred_orig = dataset.inverse_normalize_volume(y_pred)
            y_batch_orig = dataset.inverse_normalize_volume(batch[target_idx])

            if norm_by_dayly_deviation:
                factor = batch[volume_journalier_moyen_idx].unsqueeze(1)
                y_pred_orig = y_pred_orig * factor
                y_batch_orig = y_batch_orig * factor

            _metrics_track.update(y_pred_orig, y_batch_orig)

            predicted_values.append(y_pred_orig.cpu().detach().to(torch.float32).numpy().reshape(-1))
            if i < dataloader_length - 1:  #Append seulement 1 valeur réelle pour chaque batch sauf le dernier
                real_values.append(y_batch_orig.cpu().detach().to(torch.float32).numpy()[0,0,0])
            else:  # Append toutes les valeurs réelles pour le dernier batch
                real_values.extend(y_batch_orig.cpu().detach().to(torch.float32).numpy().reshape(-1))

        _metrics = _metrics_track.compute()

    return real_values, predicted_values, _metrics

def render_sap_graph_H1_with_data(real_values: list, predicted_values: list, _metrics: dict, figsize=(12, 6), dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(real_values, label="Valeurs réelles", marker="o")
    ax.plot(predicted_values, label="Valeurs prédites", marker="x")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Volume (litres)")
    ax.set_title("Real vs Predicted Values")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),  # same x, higher y
        borderaxespad=0
    )
    return fig, ax, _metrics

def render_sap_graph_H1(model: BaseModel, dataset: ACERDataset, figsize=(12, 6), dpi=200):
    real_values, predicted_values, _metrics = __get_prediction_and_real_values(model, dataset)
    return render_sap_graph_H1_with_data(real_values, predicted_values, _metrics, figsize=figsize, dpi=dpi)

def render_sap_graph_Hn_with_data(real_values: list, predicted_values: list, _metrics: dict, horizon: int, figsize=(12, 6), dpi=200):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Valeurs réelles
    ax.plot(
        range(len(real_values)),
        real_values,
        label="Valeurs réelles",
        color="blue",
        # marker="o",
        # s=10,  # Taille du marqueur
        alpha=0.8,  # Transparence pour mieux visualiser les chevauchements
        zorder=1  # Assurez-vous que ces points sont en dessous des autres
    )

    # Valeurs prédites
    colors = ["red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan", "magenta"]

    for i in range(len(predicted_values)):
        start = i
        end = i + horizon
        x_axis = np.arange(start, end)

        color = colors[i % len(colors)]  # Cycle through colors if there are more predictions than colors
        ax.plot(
            x_axis,
            predicted_values[i],
            # label=f"Predicted Values (Batch {i+1})",
            color=color,
            marker="o",
            ms=1, # Taille du marqueur
            alpha=0.5  # Transparence pour mieux visualiser les chevauchements
        )
        ax.scatter(
            x_axis[0],
            predicted_values[i][0],
            color=color,
            marker="s",
            s=10,
            alpha=0.5
        )
    return fig, ax, _metrics

def render_sap_graph_Hn(model: BaseModel, dataset: ACERDataset, horizon: int, figsize=(12, 6), dpi=200):
    real_values, predicted_values, _metrics = __get_prediction_and_real_values(model, dataset)
    return render_sap_graph_Hn_with_data(real_values, predicted_values, _metrics, horizon, figsize=figsize, dpi=dpi)

def sap_graph_add_axes_and_metrics(fig, ax, _metrics, showDates=10, datesAngle=45, dates=None, sup_text=None, show_metrics=True):
    ax.set_title("Courbe des valeurs réelles et prédites")
    ax.set_ylabel("Volume/entailles (litres)", fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)  # Ajouter une grille pour l'axe y

    if show_metrics or sup_text is not None:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0),  # same x, higher y
            borderaxespad=0
        )
    else:
        ax.legend(loc="best")

    if showDates > 0 and dates is not None:
        ax.set_xlabel("Date", fontweight='bold')
        # ax.set_xticks(range(0, len(dates), showDates))
        # ax.set_xticklabels(
        #     [str(d)[:10] for d in dates[::showDates]],
        #     rotation=datesAngle,
        #     ha="right",
        # )

        # Trouver les débuts de courbe en detectant les NaN dans dates
        curve_start_indices = [0]  # Le premier point est toujours un début
        for i in range(1, len(dates)):
            if dates[i-1]=="" and dates[i]!="":
                curve_start_indices.append(i)


        # Créer les ticks à partir de chaque début de courbe
        tick_positions = []
        tick_labels = []

        for i, curve_start in enumerate(curve_start_indices):
            # Fin de cette courbe = début de la prochaine (ou fin du tableau)
            curve_end = curve_start_indices[i + 1] if i + 1 < len(curve_start_indices) else len(dates)

            # Ajouter les ticks tous les showDates à partir du début de cette courbe
            for idx in range(curve_start, curve_end, showDates):
                if idx < len(dates):
                    tick_positions.append(idx)
                    tick_labels.append(str(dates[idx])[:10])

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=datesAngle, ha="right")


        ax.grid(axis='x', linestyle='--', alpha=0.5)  # Ajouter une grille pour l'axe x

    if show_metrics: # Afficher les métriques sur le graphique
        metrics_text = (
            "        Metrics\n"
            "-------------------------\n"
            f"MSE:        {_metrics['MSE']:.4f}\n"
            f"RMSE:      {_metrics['RMSE']:.4f}\n"
            f"RMSE%:   {_metrics['RMSE%']:.2f}%\n"
            f"WAPE:      {_metrics['WAPE']:.2f}%\n"
            f"R2:          {_metrics['R2']:.4f}"
        )
        # Add metrics in top-left corner
        ax.text(
            1.05, 0.0,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="lightgray",
                edgecolor="steelblue",
                alpha=0.5
            )
        )

    if sup_text is not None:
        ax.text(
            1.05, 0.5,
            sup_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(
                boxstyle="round",
                facecolor="lightgray",
                edgecolor="steelblue",
                alpha=0.5
            )
        )

    return fig, ax

def sap_graph(model: BaseModel, dataset: ACERDataset, horizon: int, figsize=(12, 6), dpi=200, showDates=10, datesAngle=45, sup_text=None, show_metrics=True, metrics: dict|None=None, title:str|None=None):
    id_date_sys = dataset._resolve_field("id_date_system")
    dates = id_date_sys[:, 0]

    real_values, predicted_values, _metrics = __get_prediction_and_real_values(model, dataset)

    metrics_passed = metrics if metrics is not None else _metrics

    return sap_graph_with_data(
        real_values = real_values,
        predicted_values = predicted_values,
        horizon = horizon,
        dates = dates,
        figsize = figsize,
        dpi = dpi,
        showDates = showDates,
        datesAngle = datesAngle,
        sup_text = sup_text,
        show_metrics = show_metrics,
        metrics = metrics_passed,
        title = title
    )

    # if horizon == 1:
    #     fig, ax, _metrics = render_sap_graph_H1(model, dataset, figsize=figsize)
    # else:
    #     fig, ax, _metrics = render_sap_graph_Hn(model, dataset, horizon, figsize=figsize)

    # fig, ax = sap_graph_add_axes_and_metrics(fig, ax, _metrics, showDates=showDates, datesAngle=datesAngle, dates=dates, sup_text=sup_text, show_metrics=show_metrics)
    # if title is not None:
    #     ax.set_title(title, fontsize=14, fontweight='bold')

    # plt.close(fig)  # Fermer la figure pour éviter l'affichage dans les notebooks

    # return fig, ax

def sap_graph_with_data(real_values: list, predicted_values: list, horizon: int, dates: np.ndarray|None = None, figsize=(12, 6), dpi=200, showDates=10, datesAngle=45, sup_text=None, show_metrics=True, metrics: dict|None=None, title:str|None=None):
    if metrics is None:
        rv_tensor = torch.tensor(real_values)
        pv_tensor = torch.tensor(predicted_values)
        mask = torch.isfinite(rv_tensor) & torch.isfinite(pv_tensor)

        _metrics_track = RegressionMetrics(device="cpu")
        _metrics_track.update(pv_tensor[mask], rv_tensor[mask])
        _metrics = _metrics_track.compute()
    else:
        _metrics = metrics

    # Inserer NaN valeurs entre les date discontinues pour séparer les courbes
    if dates is not None:
        # trouver les indices où les dates sont discontinues
        time_diffs = np.diff(dates)
        discontinuous_indices = np.where(time_diffs > np.timedelta64(1, 'D'))[0] + 1
        for idx in reversed(discontinuous_indices):
            for _ in range(horizon + 1):  # Insérer NaN pour chaque horizon sauf le dernier
                real_values.insert(idx, np.nan)
                predicted_values.insert(idx, np.full((horizon), np.nan))
                dates = np.insert(dates, idx, "")

    if horizon == 1:
        fig, ax, _metrics = render_sap_graph_H1_with_data(real_values, predicted_values, _metrics, figsize=figsize, dpi=dpi)
    else:
        fig, ax, _metrics = render_sap_graph_Hn_with_data(real_values, predicted_values, _metrics, horizon, figsize=figsize, dpi=dpi)

    fig, ax = sap_graph_add_axes_and_metrics(fig, ax, _metrics, showDates=showDates, datesAngle=datesAngle, dates=dates, sup_text=sup_text, show_metrics=show_metrics)
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.close(fig)  # Fermer la figure pour éviter l'affichage dans les notebooks

    return fig, ax