from torchvision.models.resnet import ResNet, BasicBlock,resnet152
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import torch
from torch import nn, optim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from torch.utils.data import DataLoader
from Natasha1 import Natasha1
from Natasha2 import Natasha2
from Natasha2_hess_prod import Natasha2_hp
from models import *
from utils import *
import random

def train_val(algorithm='Natasha2', cuda=0, net='MnistLeNet',
                    epochs=30, train_portion=0.1, train_batch=64,
                    val_batch=64, dataset='MNIST'):
    ''' Wrapper method to perform 1 experiment on Mnist digit classification
    optim        : optimization algorithm to used ['Natasha2', 'Natasha1', 'Adam', 'SGD', 'SGD_momentum']
    cuda         : which cuda device to use
    net          : which net to use ['MnistLeNet', 'MnistResNet', 'CifarLeNet', 'CifarResNet']
    train_portion: portion of training dataset to use
    '''
    natasha1_param = {'ALPHA': 0.01, 'B': 27, 'P': 9, 'SIGMA': 0}
    natasha2_param = {'ALPHA': 0.01, 'B': 27, 'P': 9, 'SIGMA': 0.01, 'DELTA': 0.05, 'ETA': 0.1}
    
    start_ts = time.time()
    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % cuda)
        print('[train.py] Using cuda device %d' % cuda)
    else:
        device = torch.device('cpu')
        print('[train.py] Using CPU...')

    if net == 'MnistLeNet':
        model = MnistLeNet().to(device)
    elif net == 'MnistResNet':
        model = MnistResNet().to(device)
    elif net == 'CifarLeNet':
        model = CifarLeNet().to(device)
    elif net == 'CifarResNet':
        model = CifarResNet().to(device)
        

    train_loader, val_loader = get_data_loaders(train_batch, val_batch, dataset=dataset)

    if algorithm == 'Natasha2':
        optimizer = Natasha2(model.parameters(), alpha=natasha2_param['ALPHA'], B=natasha2_param['B'],
                             p=natasha2_param['P'], sigma = natasha2_param['SIGMA'], 
                             delta=natasha2_param['DELTA'], eta =natasha2_param['ETA'])
    elif algorithm == "Natasha2_hp":
        optimizer = Natasha2_hp(model.parameters(), alpha=natasha2_param['ALPHA'], B=natasha2_param['B'],
                             p=natasha2_param['P'], sigma = natasha2_param['SIGMA'], 
                             delta=natasha2_param['DELTA'], eta =natasha2_param['ETA'])
    elif algorithm == 'Natasha1':
        optimizer = Natasha1(model.parameters(), alpha=natasha1_param['ALPHA'], B=natasha1_param['B'],
                              p=natasha1_param['P'], sigma = natasha1_param['SIGMA'])
    elif algorithm == 'Adam':
        optimizer = optim.Adam(model.parameters())
    elif algorithm == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr = 0.01)
    elif algorithm == 'SGD_momentum':
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    
    print('[train.py] using optimization algorithm %s' % algorithm)
    print('[train.pu] training with %d%% of data' % int(train_portion * 100))
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss() 

    losses = []
    batches = len(train_loader)
    val_batches = len(val_loader)

    train_loss      = []
    validation_loss = []
    precision_list  = []
    recall_list     = []
    F1_list         = []
    accuracy_list   = []
    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):

        total_loss = 0

        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  -------------------- 
        # set model to training
        model.train()

        for i, data in progress:
            if random.random() < train_portion:
                X, y = data[0].to(device), data[1].to(device)

                # training step for single batch
                model.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward(retain_graph=True)
                if algorithm == 'Natasha2':
                    optimizer.step(eval_hessian(loss, model))
                if algorithm == 'Natasha2_hp':
                    grads_0 = torch.cat([param.grad.view(-1) for param in model.parameters()])
                    kick_criterion, v = oja_criterion(optimizer.delta, grads_0,model, X, y, criterion, eta = optimizer.eta)
                    optimizer.step(model, kick_criterion, v)
                else:
                    optimizer.step()

                # getting training quality data
                total_loss += loss.item()

                # updating progress bar
                progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

        # releasing unceseccary memory in GPU
        torch.cuda.empty_cache()

        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        precision, recall, f1, accuracy = [], [], [], []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data[0].to(device), data[1].to(device)

                outputs = model(X)
                val_losses += criterion(outputs, y)
                predicted_classes = torch.max(outputs, 1)[1] # get class from network's prediction

                # calculate P/R/F1/A metrics for batch
                for acc, metric in zip((precision, recall, f1, accuracy), 
                                       (precision_score, recall_score, f1_score, accuracy_score)):
                    acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()))

        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")
        train_loss.append(total_loss/batches)
        validation_loss.append(val_losses/val_batches)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(f1)
        accuracy_list.append(accuracy)
        print_scores(precision, recall, f1, accuracy, val_batches)
        losses.append(total_loss/batches) # for plotting learning curve
    print(f"Training time: {time.time()-start_ts}s")
    learning_curves = {'train_loss': train_loss, 'validation_loss': validation_loss, 'precision': precision_list,
                       'recall': recall_list, 'F1': F1_list, 'accuracy': accuracy_list}
    return learning_curves
                       
                       
                       
                       
                       