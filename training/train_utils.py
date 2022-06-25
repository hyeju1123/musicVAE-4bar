import numpy as np
import torch
import torch.nn.functional as F


def inverse_sigmoid(epoch, k=20):
    return k / (k + np.exp(epoch/k))


def kl_annealing(epoch, start, end, rate=0.9):
    return end + (start - end)*(rate)**epoch


def vae_loss(recon_x, x, mu, std, beta=0):
    logvar = std.pow(2).log()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + (beta * KLD)


def accuracy(y_true, y_pred):
    y_true = torch.argmax(y_true, axis=2)
    total_num = y_true.shape[0] * y_true.shape[1]

    return torch.sum(y_true == y_pred) / total_num


def prob_label(prob):
    num_classes = prob.shape[1]
    play = np.zeros(prob.shape)

    for seq in range(prob.shape[0]):
        label = np.random.choice(num_classes, size=1, p=prob[seq])
        play[seq, label] = 1

    return play