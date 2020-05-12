import torch
from torchvision.datasets import MNIST, CIFAR10
from torch import nn, optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torch.utils.data import DataLoader
import inspect

# eval Hessian matrix
def eval_hessian(loss, model):
    loss_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    del loss_grad, grad2rd, g2
    hessian = hessian.detach()
    torch.cuda.empty_cache()
    return hessian

def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)

def print_scores(p, r, f1, a, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def get_data_loaders(train_batch_size, val_batch_size, dataset='MNIST'):
    if dataset == 'MNIST':
        mnist = MNIST(download=True, train=True, root=".").train_data.float()

        data_transform = Compose([Resize(32, interpolation=2), ToTensor()])

        train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                                  batch_size=train_batch_size, shuffle=True, num_workers=16)

        val_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=False),
                                batch_size=val_batch_size, shuffle=False, num_workers=16)
    elif dataset == 'CIFAR':
        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=True, num_workers=16)

        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        val_loader = DataLoader(testset, batch_size=val_batch_size,
                                         shuffle=False, num_workers=16)
    return train_loader, val_loader

def oja_criterion(delta, grad_0,model, X, y, criterion, eta):
    n_param = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_param,dtype = torch.float32, device = "cuda")
    for i in range(int(1/(delta ** 2))):
        v = torch.flatten(torch.mm(torch.ones((n_param,n_param)), v)) - eta * hessian_w_approx(model,X,y,criterion,v, grad_0)
        v = v / torch.norm(v)
    kick_criterion = torch.dot(v, hessian_w_approx(model, X,y, criterion, v, grad_0))
    return kick_criterion, v

def hessian_w_approx(model, X, y, criterion, v, grad_0 ,q = 0.0001):
    model_copy = type(model)().cuda() # get a new instance
    model_copy.load_state_dict(model.state_dict())
    v_update(model_copy, v, q)
    grad_1 = get_grad(model_copy, X, y, criterion)
    return (grad_1 - grad_0)/q

def v_update(model, v, q):
    vec = torch.nn.utils.parameters_to_vector(model.parameters())
    vec.add_(v, alpha = q)
    torch.nn.utils.vector_to_parameters(vec, model.parameters())

def get_grad(model, X, y, criterion):
    model.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward(retain_graph=True)
    grads = torch.cat([param.grad.view(-1) for param in model.parameters()])
    return grads
