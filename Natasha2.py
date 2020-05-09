import torch
from torch.distributions.bernoulli import Bernoulli
from utils import *


class Natasha2(torch.optim.Optimizer):
    def __init__(self, params, alpha=0.001, B=100, p=5, sigma=1, delta=0.1, eta=0.01):
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if B < 0.0:
            raise ValueError("Invalid B value: {}".format(B))
        if p < 0.0:
            raise ValueError("Invalid p value: {}".format(p)) 
        if sigma < 0.0:
            raise ValueError("Invalid sigma value: {}".format(sigma)) 
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta)) 
        if eta < 0.0:
            raise ValueError("Invalid eta value: {}".format(eta)) 
        self.delta = delta
        self.eta = eta
        defaults = dict(params=params, alpha=alpha, B=B, p=p, sigma=sigma, delta=delta, eta=eta)
        super(Natasha2, self).__init__(params, defaults)
        self.bern = Bernoulli(torch.tensor([0.5]))

    def __setstate__(self, state):
        super(Natasha1, self).__setstate__(state)

    @torch.no_grad()
    def step(self, hessian_matrix):
        hess = -hessian_matrix
        v = torch.rand(hess.size()[0],1)
        Mi = torch.ones_like(hess) + self.eta * hess
        for i in range(int(1/(self.delta ** 2))):
            v = torch.matmul(Mi, v)
        v = v / torch.norm(v)
        if (torch.chain_matmul(v.transpose(0,1), hess, v) <= -self.delta /2):
            print('kick')
            kick = self.bern.sample()*2-1
            for group in self.param_groups:
                for ctr, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    p.add_(v[ctr], alpha = kick*group['delta'])
        else:
            print('not kick')
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.view(-1)
                    x_h = p.view(-1)
                    for s in range(group['p']):
                        x_0 = x_h
                        x_list = []
                        x_list.append(x_0)
                        for t in range(int(group['B']/group['p'])):
                            d_tmp = d_p + 2*group['sigma']*(x_list[t] - x_h)
                            x_list.append(x_list[t] - group['alpha'] * d_tmp)
                        temp = torch.stack(x_list)
                        x_h = torch.mean(temp, dim=0)
                    x_h = x_h.reshape(p.shape)

                    p.eq_(x_h)

