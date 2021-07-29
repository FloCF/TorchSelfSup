import torch
from torch.optim import Optimizer

class LARS(Optimizer):
    """
    From: https://github.com/facebookresearch/vissl/blob/master/vissl/optimizers/lars.py
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eta: float = 0.001,
        exclude_bias_and_norm: bool = False,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eta": eta,
            "exclude": self._exclude_bias_and_norm if exclude_bias_and_norm else None,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _exclude_bias_and_norm(p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                dp = dp.add(p, alpha=g["weight_decay"])

                if g["exclude"] is None or not g["exclude"](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)

                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])