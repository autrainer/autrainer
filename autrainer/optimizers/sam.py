from typing import Callable, Tuple

import torch

import autrainer


class SAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: torch.nn.Module,
        base_optimizer: str,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        """Sharpness Aware Minimization (SAM) optimizer.

        This implementation is adapted from the following repository:
        https://github.com/davda54/sam

        For more information, see:
        https://arxiv.org/abs/2010.01412

        Args:
            params: Model parameters.
            base_optimizer: Underlying optimizer performing the sharpness-aware
                update, specified as a relative import path.
            rho: Size of the neighborhood for computing the max loss.
                Defaults to 0.05.
            adaptive: Whether to use an experimental implementation of
                element-wise Adaptive SAM. Defaults to False.
            **kwargs: Additional arguments passed to the underlying optimizer.
        """
        if rho < 0.0:
            raise ValueError(f"Rho '{rho}' should be non-negative.")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = autrainer.instantiate(
            config={
                "_target_": base_optimizer,
                "params": self.param_groups,
                **kwargs,
            },
            instance_of=torch.optim.Optimizer,
        )
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    # @torch.no_grad()
    # def first_step_uphill(self, zero_grad=False):
    #     grad_norm = self._grad_norm()
    #     for group in self.param_groups:
    #         scale = group["rho"] / (grad_norm + 1e-12)

    #         for p in group["params"]:
    #             if p.grad is None: continue
    #             self.state[p]["old_p"] = p.data.clone()
    #             e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
    #             p.add_(e_w)  # climb to the local maximum "w + e(w)"

    #     if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # get back to "w" from "w + e(w)"
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        # the closure should do a full forward-backward pass
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    # ? Training Tips functions from the Github repo ref: https://github.com/davda54/sam
    # TODO: Check if Batch Norm Tip needs to be applied
    def custom_step(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
        probabilities_fn: Callable,
    ) -> Tuple[float, torch.Tensor]:
        """Sharpness Aware Minimization requires two forward-backward passes
        over the batch to simultaneously minimize the loss value and the loss
        sharpness.

        Args:
            model: Model to be optimized.
            data: Batched input data.
            target: Batched target data.
            criterion: Loss function.
            probabilities_fn: Function to get probabilities from model outputs.

        Returns:
            Reduced loss over the batch and detached model outputs.
        """
        output = model(data)
        loss = criterion(probabilities_fn(output), target)
        loss.backward()
        self.first_step(zero_grad=True)

        output = model(data)
        loss = criterion(probabilities_fn(output), target)
        loss.backward()
        self.second_step(zero_grad=True)
        _loss = loss.item()
        return _loss, output.detach()
