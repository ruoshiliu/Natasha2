
#  Natasha2

Implementation of Natasha1.5 and Natasha2 algorithm in PyTorch

Natasha1.5 & Natasha2 algorithm in Pytorch

Full technical report: [technical_report.pdf](images/technical_report.pdf)

This repo contains a PyTorch implementation of Natasha2 from the paper "[Natasha 2: Faster Non-Convex Optimization Than SGD](https://arxiv.org/abs/1708.08694)" by Zeyuan Allen-Zhu.

To run experiments in MNIST or CIFAR-10, please use notebook [MNIST.ipynb](MNIST.ipynb) and [CIFAR.ipynb](CIFAR.ipynb)

##  Experimental results

  We tested both Natasha1.5 and Natasha2 on two models -- Mini-LeNet-4 and ResNet-18 (see [models.py](models.py) for architecture details).

Mini-LeNet-4 contains only ~600 parameters and ResNet-18 contains standard 11M parameters. We constructed a mini version of LeNet-4 because Natasha2 requires a second order hessian matrix which is extremely computationally expensive. Thus, given our hardware resources, we are able to train a Mini-LeNet-4 with reasonable period of training time.

Note: Mini-LeNet-4 doesn't have enough capacity to classify CIFAR-10, thus having strange learning curves


###  Learning Curves

####  ResNet-18 trained on CIFAR-10

![Alt text](images/CIFAR10_ResNet.png?raw=true "Title")


####  ResNet-18 trained on MNIST

![Alt text](images/MNIST_ResNet.png?raw=true "Title")


####  Mini-LeNet-4 trained on CIFAR-10

![Alt text](images/CIFAR10_LeNet.png?raw=true "Title")


####  Mini-LeNet-4 trained on MNIST

![Alt text](images/MNIST_LeNet.png?raw=true "Title")
