"""
Orthogonal Gradient Descent Optimizer Wrapper.

This module implements a wrapper optimizer that projects gradients to be orthogonal
to the current parameters before performing an update. This can help with
optimization stability and exploration of the loss landscape.

Example usage:
    model = MyModel()
    base_optimizer = torch.optim.AdamW
    base_args = dict(lr=1e-3, weight_decay=0.01)
    optimizer = OrthoGrad(model.parameters(), base_optimizer, **base_args)
"""

import torch
from typing import Iterable, Type, Optional, Callable


class OrthoGrad(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.SGD,
        eps: float = 1e-30,
        rescale: bool = True,
        **base_optimizer_args
    ):
        """
        A wrapper optimizer that projects gradients to be orthogonal
        to the current parameters before performing an update.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer_cls (Optimizer class): The base optimizer class
                (e.g., torch.optim.SGD, torch.optim.AdamW).
            eps (float): Small constant for numerical stability.
            rescale (bool): Whether to rescale orthogonalized gradients
                to match original gradient norm.
            **base_optimizer_args: Arguments for the base optimizer.
                For example, lr=1e-3, weight_decay=1e-2, etc.
        """
        defaults = dict(eps=eps, rescale=rescale)
        super().__init__(params, defaults)

        # Create the wrapped/base optimizer using our param_groups
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)

    @torch.no_grad()
    def _orthogonalize_gradients(self, params: Iterable[torch.nn.Parameter], eps: float, rescale: bool):
        """
        Projects the gradient g to be orthogonal to the current weights w.
        
        Formula: g_orth = g - ((w·g)/(w·w + eps)) * w

        Args:
            params: Iterator over parameters
            eps: Small constant for numerical stability
            rescale: Whether to rescale orthogonalized gradients
        """
        for p in params:
            if p.grad is None:
                continue

            # Reshape tensors to 1D for simpler computation
            w = p.view(-1)
            g = p.grad.view(-1)

            # Compute orthogonal projection
            w_norm_sq = torch.dot(w, w) + eps
            proj = torch.dot(w, g) / w_norm_sq
            g_orth = g - proj * w

            if rescale:
                # Rescale to match original gradient norm
                g_norm = g.norm(2)
                g_orth_norm = g_orth.norm(2) + eps
                g_orth = g_orth * (g_norm / g_orth_norm)

            # Copy back to original gradient tensor shape
            p.grad.copy_(g_orth.view_as(p.grad))

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # First orthogonalize all gradients
        for group in self.param_groups:
            self._orthogonalize_gradients(
                group['params'],
                group['eps'],
                group['rescale']
            )

        # Then let base optimizer do its update
        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clears the gradients of all optimized parameters.

        Args:
            set_to_none (bool): whether to set gradients to None instead
                of zeroing them.
        """
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        """
        Returns the state of the optimizer as a dict.
        """
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        
        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to state_dict().
        """
        base_state_dict = state_dict.pop('base_optimizer')
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(base_state_dict) 