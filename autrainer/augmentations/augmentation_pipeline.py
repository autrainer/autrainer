from typing import Any, Dict, List, Union

import autrainer
from autrainer.transforms import SmartCompose

from .abstract_augmentation import AbstractAugmentation
from .utils import assign_seeds, convert_shorthand


class AugmentationPipeline:
    def __init__(
        self,
        pipeline: List[Union[str, Dict[str, Any]]],
        generator_seed: int = 0,
        increment: bool = True,
    ) -> None:
        """Initialize an augmentation pipeline.

        Args:
            pipeline: The list of augmentations to apply.
            generator_seed: Seed to pass to each augmentation for
                reproducibility if the augmentation does not have a seed.
                Defaults to 0.
            increment: Whether to increment the generator seed for each
                augmentation that does not define its own seed.
                Defaults to True.
        """
        self.generator_seed = generator_seed
        self.pipeline = []
        current_seed = generator_seed
        for aug in pipeline:
            aug = convert_shorthand(aug)
            aug, current_seed = assign_seeds(aug, current_seed, increment)
            self.pipeline.append(aug)

        self.pipeline: List[AbstractAugmentation] = [
            autrainer.instantiate_shorthand(aug, instance_of=AbstractAugmentation)
            for aug in self.pipeline
        ]

    def create_pipeline(self) -> SmartCompose:
        """Create the composed and ordered augmentation pipeline.

        Returns:
            Composed augmentation pipeline.
        """
        return SmartCompose(self.pipeline)
