from typing import Any, Dict, Tuple, Union


def convert_shorthand(aug: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert shorthand syntax to a dictionary.

    Args:
        aug: The augmentation to convert.

    Returns:
        The converted augmentation.
    """
    if isinstance(aug, str):
        return {aug: {}}
    return aug


def assign_seeds(
    aug: Dict[str, Any],
    current_seed: int,
    increment: bool,
) -> Tuple[Dict[str, Any], int]:
    """Recursively assign optionally incrementing seeds to augmentation
    configurations.

    Args:
        aug: Potentially nested augmentation configuration.
        current_seed: The current seed to assign.
        increment: Whether to increment the seed after assignment. If True,
            the seed of each augmentation will be incremented by 1.
            If False, the same seed will be used for all augmentations.

    Returns:
        The updated augmentation configuration and the next seed to use.
    """
    aug_name = next(iter(aug))
    config = aug[aug_name]

    if config.get("generator_seed") is None:
        config["generator_seed"] = current_seed
        if increment:
            current_seed += 1

    # recursively assign unique seeds to each augmentation in choice or
    # sequential to avoid shared seeds, which can lead to identical
    # augmentations within a batch
    if aug_name == "autrainer.augmentations.Choice" and "choices" in config:
        new_choices = []
        for choice in config["choices"]:
            choice = convert_shorthand(choice)
            updated_choice, current_seed = assign_seeds(choice, current_seed, increment)
            new_choices.append(updated_choice)
        config["choices"] = new_choices

    elif aug_name == "autrainer.augmentations.Sequential" and "sequence" in config:
        new_sequence = []
        for seq_aug in config["sequence"]:
            seq_aug = convert_shorthand(seq_aug)
            updated_aug, current_seed = assign_seeds(seq_aug, current_seed, increment)
            new_sequence.append(updated_aug)
        config["sequence"] = new_sequence

    return {aug_name: config}, current_seed
