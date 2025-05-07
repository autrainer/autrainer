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
    aug_name = next(iter(aug))
    config = aug[aug_name]

    if "generator_seed" not in config or config["generator_seed"] is None:
        config["generator_seed"] = current_seed
        if increment:
            current_seed += 1

    if aug_name == "autrainer.augmentations.Choice" and "choices" in config:
        new_choices = []
        for choice in config["choices"]:
            choice = convert_shorthand(choice)
            updated_choice, current_seed = assign_seeds(
                choice, current_seed, increment
            )
            new_choices.append(updated_choice)
        config["choices"] = new_choices

    elif (
        aug_name == "autrainer.augmentations.Sequential"
        and "sequence" in config
    ):
        new_sequence = []
        for seq_aug in config["sequence"]:
            seq_aug = convert_shorthand(seq_aug)
            updated_aug, current_seed = assign_seeds(
                seq_aug, current_seed, increment
            )
            new_sequence.append(updated_aug)
        config["sequence"] = new_sequence

    return {aug_name: config}, current_seed
