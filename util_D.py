from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_network(net, name):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number parameters of {} : {}'. format(name, num_params))


def load_pretrained_model(model, pretrained_dict, wfc=True):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    if wfc:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ((k in model_dict) and ('fc' not in k))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


def save_checkpoint(state, filename):
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # transposition
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cpu_gpu(use_gpu, data_tensor, volatile=False):
    if use_gpu:
        data = Variable(data_tensor.cuda(), volatile=volatile)
    else:
        data = Variable(data_tensor, volatile=volatile)
    return data


def pkt_cosine_similarity_loss(output_s, output_t, eps=1e-5):
    # out_s: (16, 3, 32, 32)
    # out_t: (16, 3, 32, 32)
    # Normalize each vector by its norm
    # output_s/output_t: (batch_size, num_class)
    output_s = output_s.view(output_s.size(0), -1)  # (16, 3*32*32) [-1, 1]
    output_s_norm = torch.sqrt(torch.sum(output_s ** 2, dim=1, keepdim=True))  # (16, 1)
    output_s = output_s / (output_s_norm + eps)   # Normalization  Add Xi**2 in each row up to 1.
    output_s[output_s != output_s] = 0  #

    output_t = output_t.view(output_t.size(0), -1)
    output_t_norm = torch.sqrt(torch.sum(output_t ** 2, dim=1, keepdim=True))
    output_t = output_t / (output_t_norm + eps)
    output_t[output_t != output_t] = 0  #

    # Calculate the cosine similarity
    output_s_cos_sim = torch.mm(output_s, output_s.transpose(0, 1)) # (16, 16)
    output_t_cos_sim = torch.mm(output_t, output_t.transpose(0, 1))

    # Scale cosine similarity to [0,1]
    output_s_cos_sim = (output_s_cos_sim + 1.0) / 2.0
    output_t_cos_sim = (output_t_cos_sim + 1.0) / 2.0

    # Transform them into probabilities
    output_s_cond_prob = output_s_cos_sim / torch.sum(output_s_cos_sim, dim=1, keepdim=True)
    output_t_cond_prob = output_t_cos_sim / torch.sum(output_t_cos_sim, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = torch.mean(output_t_cond_prob * torch.log((output_t_cond_prob + eps) / (output_s_cond_prob + eps)))

    return loss


if __name__ == '__main__':
    topk = (1, 3)
    maxk = max(topk)
    print(max(topk))

    output = torch.LongTensor([[5, 0, 8, 7],
                               [4, 1, 4, 6],
                               [3, 6, 3, 0],
                               [7, 3, 5, 4]])

    target = torch.LongTensor([[2],
                               [0],
                               [2],
                               [3]])
    print(output)
    _, pred = output.topk(maxk, 1, True, True)
    print('Pred', pred)
    pred = pred.t()
    print('pred2', pred)
    print("----------------------")
    print(target.view(1, -1))
    target = target.view(1, -1).expand_as(pred)
    print('target:', target)
    correct = pred.eq(target)
    print('correct:', correct)
    res = []
    for k in topk:
        print('k:', k)
        correct_k = correct[:k]
        print('correct_k:', correct_k)
        # print(correct_k.mul_(100.0 / 64.0))
