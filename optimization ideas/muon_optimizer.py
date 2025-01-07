import torch
from torch.optim import Optimizer
import math

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to orthogonalize G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()  # you can also do .half() or float32
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm <= 1
    X = X / (X.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    X = (G.T.type_as(X) @ X).float().trace().bfloat16() * X  # Adaptive scaling
    return X

class Muon(Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz for 2D parameters only
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(d_p, alpha=1.0)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                # for 2D param, orthogonalize the update
                if p.ndim == 2:
                    d_p = zeropower_via_newtonschulz5(d_p, ns_steps)

                # param update
                p.data.add_(d_p, alpha=-lr)

        return None 