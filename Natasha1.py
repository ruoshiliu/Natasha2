import torch

class Natasha1(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.001, B=100, p=5, sigma=1):
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if B < 0.0:
            raise ValueError("Invalid B value: {}".format(B))
        if p < 0.0:
            raise ValueError("Invalid p value: {}".format(p))
        if sigma < 0.0:
            raise ValueError("Invalid p value: {}".format(sigma))

        defaults = dict(params=params, alpha=alpha, B=B, p=p, sigma=sigma)
        super(Natasha1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Natasha1, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                x_h = p
                for s in range(group['p']):
                    x_list = [x_h]
                    for t in range(int(group['B']/group['p'])):
                        d_tmp = d_p + group['sigma']*(x_list[t] - x_h)
                        x_list.append(x_list[t].add_(d_tmp, alpha = -group['alpha']))
                    temp = torch.stack(x_list)
                    x_h = torch.mean(temp, dim=0)
                p=x_h
