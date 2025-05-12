import os
import torch
import mlflow
import mlflow.pytorch
from pathlib import Path

from data.data import MVTecDataset, DEFAULT_SIZE
from model.patch_core import PatchCore
from utils.utils import backbones, dataset_scale_factor


def train_and_log(
    cls: str,
    backbone_key: str = "WideResNet50",
    f_coreset: float = 0.1,
    eps: float = 0.9,
    k_nn: int = 3,
    use_cache: bool = True,
    cache_root: str = "./patchcore_cache",
    mlflow_experiment: str = "PatchCore_MVTec"
) -> dict:

    f_coreset = float(f_coreset)
    eps       = float(eps)
    k_nn      = int(k_nn)
    if not isinstance(use_cache, bool):
        use_cache = str(use_cache).lower() in ("true", "1", "yes")

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run() as run:

        # Choix du backbone
        vanilla = (backbone_key == "WideResNet50")
        size = DEFAULT_SIZE if vanilla else {
            'ResNet50': 224,
            'ResNet50-4': 288,
            'ResNet50-16': 384,
            'ResNet101': 224
        }[backbone_key]

        mlflow.log_params({
            "cls": cls,
            "backbone_key": backbone_key,
            "f_coreset": f_coreset,
            "eps": eps,
            "k_nn": k_nn,
            "use_cache": use_cache,
            "image_size": size
        })

        train_dl, test_dl = MVTecDataset(cls, size=size, vanilla=vanilla).get_dataloaders()

        model = PatchCore(
            f_coreset=f_coreset,
            eps_coreset=eps,
            k_nearest=k_nn,
            vanilla=vanilla,
            backbone=backbones[backbone_key],
            image_size=size
        )
        model.fit(train_dl, scale=dataset_scale_factor[backbone_key])

        cache_dir = Path(cache_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        mb = model.memory_bank
        if not isinstance(mb, torch.Tensor):
            mb = torch.cat(mb, dim=0)
        mb = mb.cpu()
        torch.save(mb, cache_dir / f"{cls}_{backbone_key}_f{f_coreset:.3f}.pth")

        # Évaluation
        image_auc, pixel_auc = model.evaluate(test_dl)

        # Log métriques
        mlflow.log_metrics({
            "image_level_roc_auc": image_auc,
            "pixel_level_roc_auc": pixel_auc
        })

        # Log du modèle complet
        mlflow.pytorch.log_model(model, artifact_path="patchcore_model")

        return {
            "image_level_roc_auc": image_auc,
            "pixel_level_roc_auc": pixel_auc,
            "run_id": run.info.run_id
        }
