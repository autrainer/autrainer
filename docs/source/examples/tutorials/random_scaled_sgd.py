from typing import Tuple

import torch


class RandomScaledSGD(torch.optim.Optimizer):
    def __init__(
        self,
        scaling_factor: float = 0.01,
        p: float = 1.0,
        generator_seed: int = None,
        *args,
        **kwargs,
    ) -> None:
        """Randomized Scaled SGD optimizer. Randomly scales the learning rate.

        Args:
            scaling_factor: Learning rate scaling factor. Defaults to 1.0.
            p: Probability of scaling the learning rate. Defaults to 1.0.
            generator_seed: Seed for the random number generator.
                Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.scaling_factor = scaling_factor
        self.p = p
        self.g = torch.Generator()
        self.base_lr = self.param_groups[0]["lr"]
        if generator_seed:
            self.g.manual_seed(generator_seed)

    def custom_step(
        self,
        model: torch.nn.Module,  # model
        data: torch.Tensor,  # batched input data
        target: torch.Tensor,  # batched target data
        criterion: torch.nn.Module,  # loss function
    ) -> Tuple[float, torch.Tensor]:
        self.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if torch.rand(1, generator=self.g).item() < self.p:
            self.param_groups[0]["lr"] *= self.scaling_factor
        self.step()
        self.param_groups[0]["lr"] = self.base_lr
        return loss.item(), output.detach()
