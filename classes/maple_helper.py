import torch

'''
Aide à la normalisation des tenseurs d'entrée et de sortie.
Utiliser par ACERDataModule
'''

class NormHelper:
    @staticmethod
    def compute_normalization_stats(tensor: torch.Tensor, robust: bool = False, per_feature: bool = True) -> dict:
            """ Calcul des statistiques de normalisation. """
            if robust:
                median = tensor.median(dim=0, keepdim=True)[0] if per_feature else tensor.median()
                q1 = tensor.quantile(0.25, dim=0, keepdim=True) if per_feature else tensor.quantile(0.25)
                q3 = tensor.quantile(0.75, dim=0, keepdim=True) if per_feature else tensor.quantile(0.75)
                iqr = (q3 - q1).clamp(min=1e-6)
                return {"center": median, "scale": iqr}
            else:
                mean = tensor.mean(dim=0, keepdim=True) if per_feature else tensor.mean()
                std = tensor.std(dim=0, keepdim=True).clamp(min=1e-6) if per_feature else tensor.std().clamp(min=1e-6)
                return {"center": mean, "scale": std}

    @staticmethod
    def apply_normalization(tensor: torch.Tensor, stats: dict) -> torch.Tensor:
        """
        Applique la normalisation aux tenseurs d'entrée et de cible en utilisant les statistiques fournies.
        """
        # Vérifie que les stats sont valides
        if 'center' not in stats or 'scale' not in stats:
            raise ValueError("Normalization stats must contain 'center' and 'scale' keys.")
        return (tensor - stats['center']) / stats['scale']

    @staticmethod
    def inverse_normalize(normalized_tensor: torch.Tensor, stats: dict) -> torch.Tensor:
        """
        Inverse la normalisation des cibles en utilisant les statistiques fournies.

        Args:
            normalized_tensor (torch.Tensor): Tenseur normalisé à inverser.
            stats (dict): Dictionnaire contenant les statistiques de normalisation avec les clés 'center' et 'scale'.
        """
        if 'center' not in stats or 'scale' not in stats:
            raise ValueError("Normalization stats must contain 'center' and 'scale' keys.")

        return normalized_tensor * stats['scale'] + stats['center']