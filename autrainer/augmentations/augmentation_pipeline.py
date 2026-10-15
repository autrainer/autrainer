from typing import Any

import autrainer
from autrainer.transforms import AbstractTransform, SmartCompose

from .abstract_augmentation import AbstractAugmentation
from .utils import assign_seeds, convert_shorthand


class AugmentationPipeline:
    def __init__(
        self,
        pipeline: list[str | dict[str, Any]],
        generator_seed: int = 0,
        increment: bool = True,
    ) -> None:
        """Initialize an augmentation pipeline.

        Args:
            pipeline: The list of augmentations to apply.
            generator_seed: Seed to pass to each augmentation for reproducibility if the
                augmentation does not have a seed. Defaults to 0.
            increment: Whether to increment the generator seed for each augmentation
                that does not define its own seed. Defaults to True.
        """
        self.generator_seed = generator_seed
        converted: list[str | dict[str, Any]] = []
        current_seed = generator_seed
        for aug in pipeline:
            aug = convert_shorthand(aug)
            aug, current_seed = assign_seeds(aug, current_seed, increment)
            converted.append(aug)

        self.pipeline: list[AbstractTransform] = [
            autrainer.instantiate_shorthand(aug, instance_of=AbstractAugmentation)
            for aug in converted
        ]

    def create_pipeline(self) -> SmartCompose:
        """Create the composed and ordered augmentation pipeline.

        Returns:
            Composed augmentation pipeline.
        """
        return SmartCompose(self.pipeline)
