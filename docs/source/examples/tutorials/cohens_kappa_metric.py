import sklearn.metrics

from autrainer.metrics import BaseAscendingMetric


class CohensKappa(BaseAscendingMetric):
    def __init__(self, weights: str) -> None:
        """Coehn's Kappa metric using `sklearn.metrics.cohen_kappa_score`.

        Args:
            weights: Weighting type for the metric in ["linear", "quadratic"].
        """
        super().__init__(
            name="cohens-kappa",
            fn=sklearn.metrics.cohen_kappa_score,
            weights=weights,
        )
