from typing import Dict, Optional


def format_results(
    results: Dict[str, float],
    results_type: str,
    training_type: str,
    iteration: Optional[int] = None,
) -> str:
    s = f"{results_type} results"
    s += f" at {training_type} {iteration}" if iteration else ""
    s += ":\n"
    max_key_len = max([len(k) for k in results.keys()])
    s += "\n".join(
        [f"{(k+':').ljust(max_key_len+1)} {v:.4f}" for k, v in results.items()]
    )
    return s
