"""
perdida_hibrida.py -- hybrid loss (focal + weighted BCE) and the checkpoint criterion.

REPRODUCES IN THE THESIS
    The training objective and the model-selection criterion of Section 4.4: the loss is
    0.7 * focal (gamma 3, alpha 0.95) + 0.3 * weighted BCE (positive weight 100), and the
    checkpoint criterion on the validation partition is AUC-PR + 0.1 * Youden's J.

PROVENANCE
    The functions and constants below are taken unchanged from the training script of the
    BiLSTM-Attention model in the working tree that produced the reported results, which
    the CNN-2D training script imported them from. That training script is not published
    here; only these definitions are, so that the CNN-2D training runs. See
    CHANGES-FROM-THESIS.md at the repository root.

AUTHORSHIP
    Johan S. Pena-Campos
    PhD thesis, "Interpretable Decision Support Systems for Early Cancer Diagnosis and
    Personalized Treatment", Universitat Politecnica de Catalunya and Pontificia
    Universidad Javeriana, 2026.

USE OF AI ASSISTANCE
    Generative AI (Claude, Anthropic) was used as a tool for code review, for verifying
    the code against the results reported in the thesis, for detecting and correcting
    errors, and for writing the documentation headers of this repository. The research
    design, the models, the experiments and the interpretation of the results are the
    author's. Every suggestion was reviewed by the author, who takes full responsibility
    for the content of this file.
"""

FOCAL_GAMMA = 3.0
FOCAL_ALPHA = 0.95
FOCAL_WEIGHT = 0.7
BCE_WEIGHT = 0.3
POS_WEIGHT = 100.0  # WBCE: peso del positivo


# ==================== LOSS ====================

def focal_loss(p, y, gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, eps=1e-7):
    p = p.clamp(eps, 1 - eps)
    pt = y * p + (1 - y) * (1 - p)
    at = y * alpha + (1 - y) * (1 - alpha)
    return -(at * (1 - pt) ** gamma * pt.log()).mean()


def weighted_bce(p, y, pos_weight=POS_WEIGHT, eps=1e-7):
    p = p.clamp(eps, 1 - eps)
    return -(pos_weight * y * p.log() + (1 - y) * (1 - p).log()).mean()


def hybrid_loss(p, y):
    return FOCAL_WEIGHT * focal_loss(p, y) + BCE_WEIGHT * weighted_bce(p, y)


# Criterio de checkpoint: AUC-PR (primario, robusto a desbalance de clases)
# + 0.1 * Youden J (desempate por sensibilidad-especificidad real). Con focal
# alpha=0.95 + threshold=0.5 la sensibilidad sola satura en 1.0 desde la
# primera epoca y no serviria como monitor.
J_WEIGHT = 0.1


def composite_score(metrics):
    """AUC-PR + J_WEIGHT * Youden's J. Mas alto es mejor.

    AUC-PR captura capacidad discriminativa a traves de thresholds (robusto
    a desbalance ~1% positivos). Youden's J = sens + spec - 1 captura el
    trade-off al operating threshold. El peso J_WEIGHT=0.1 lo hace
    desempatador, no dominante (AUC-PR rangea 0.01-0.07, J 0-0.2; J
    contribuye 0-0.02).
    """
    auc_pr = metrics.get('auc_pr', 0.0)
    sens = metrics.get('sens', 0.0)
    spec = metrics.get('spec', 0.0)
    J = sens + spec - 1.0
    return auc_pr + J_WEIGHT * J
