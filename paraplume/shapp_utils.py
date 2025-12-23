"""Utility functions for shap analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from paraplume.utils import get_logger

log = get_logger()

PLM_EMBEDDING_DIMENSIONS = {
    "ablang2": 480,
    "igT5": 1024,
    "igbert": 1024,
    "esm": 1280,
    "antiberty": 512,
    "prot-t5": 1024,
}
MAX_PDBS_TO_PLOT = 400
SHAP_SAMPLE_SIZE = 5


def compute_shap_importance(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    plm_dimensions: list[int],
    background: torch.Tensor,
) -> list[float]:
    """
    Compute relative importance of each PLM using SHAP values.

    Args:
        model: Trained prediction model
        embeddings: Input embeddings (B, D)
        plm_dimensions: List of dimension sizes for each PLM
        background: Background data for SHAP explainer

    Returns
    -------
        List of relative importance scores (sums to 1.0)
    """
    # Sample embeddings if dataset is large
    sample_size = min(SHAP_SAMPLE_SIZE, len(embeddings))
    sample_indices = torch.randperm(len(embeddings))[:sample_size]
    sampled_embeddings = embeddings[sample_indices]

    log.info("Computing SHAP values with GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(sampled_embeddings)
    shap_values = torch.tensor(shap_values).cpu().numpy()

    # Aggregate SHAP values by PLM
    plm_importance_scores = []
    start_idx = 0

    for dimension in plm_dimensions:
        end_idx = start_idx + dimension
        # Mean absolute SHAP value for this PLM's dimensions
        importance = np.abs(shap_values[:, start_idx:end_idx]).sum()
        plm_importance_scores.append(importance)
        start_idx = end_idx

    # Normalize to relative importance
    total_importance = np.sum(plm_importance_scores)
    return [score / total_importance for score in plm_importance_scores]


def plot_residue_importance(  # noqa: PLR0913
    labels: list[float],
    amino_acids: list[str],
    importance_matrix: np.ndarray,
    plm_names: list[str],
    save_path: Path | None = None,
    rotation=0,
    *,
    average=False,
):
    """Plot PLM importance across residue positions with fixed colors."""
    # Define fixed colors for each PLM
    color_map = {
        "esm": "#064A91",  # Darker blue
        "antiberty": "#F28C28",  # Bright orange
        "ablang2": "#C044C0",  # Strong magenta-purple
        "igt5": "#B22222",  # Firebrick red (lighter than esm)
        "igbert": "#FFD700",  # Gold
        "prot-t5": "#21C021",  # Bright green
    }
    label_map = {
        "ablang2": "AbLang2",
        "antiberty": "Antiberty",
        "esm": "ESM-2",
        "igt5": "IgT5",
        "igbert": "IgBert",
        "prot-t5": "Prot-T5",
    }
    labels_array = np.array(labels)
    amino_acids_array = np.array(amino_acids)
    num_residues, num_plms = importance_matrix.shape
    positions = np.arange(num_residues)

    _, (ax_labels, ax_importance) = plt.subplots(
        2, 1, figsize=(18, 7), sharex=True, gridspec_kw={"height_ratios": [1, 4]}
    )

    # Top panel
    ax_labels.step(positions, labels_array, where="mid", linewidth=2, color="black")
    ax_labels.set_ylim(-0.1, 1.1)
    ax_labels.set_ylabel("Prediction")
    ax_labels.grid(alpha=0.3)

    # Bottom panel: PLM importance
    for plm_idx in range(num_plms):
        key = plm_names[plm_idx].lower()
        color = color_map.get(key, None)
        label = label_map.get(key, plm_names[plm_idx])

        ax_importance.step(
            positions,
            importance_matrix[:, plm_idx],
            label=label,
            where="mid",
            linewidth=2,
            color=color,
        )

    xlabel = "Residue Position" if average else "Residue"
    ax_importance.set_ylabel("PLM Importance")
    ax_importance.set_xlabel(xlabel)
    ax_importance.grid(alpha=0.3)
    ax_importance.legend(
        title="PLM",
        title_fontsize=24,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )

    ax_importance.set_xticks(positions)
    xtick_labels = [aa for i, aa in enumerate(amino_acids_array)]
    ax_importance.set_xticklabels(xtick_labels, rotation=rotation)
    ax_importance.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    ax_labels.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        log.info("Heatmap saved.", save_path=save_path)

    plt.show()
