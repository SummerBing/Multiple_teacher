from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import sys
import argparse
import os
import time
import torchvision.datasets as datasets

# sys.path.append("/home/jtcai/SharedSSD/myCode/Multiple_teacher")

from util_D import AverageMeter, accuracy, transform_time, save_checkpoint, cpu_gpu
from vgg_network import vgg19

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
parser = argparse.ArgumentParser(description='DML on ImageNet')
python_name = 'vgg19'
save_base_root_check = './vgg19-imagenet-testrun'
parser.add_argument('--name', default='VGG19')
parser.add_argument('--epochs', type=int, default=90, help='number of total epochs to run')
parser.add_argument('--epoch_list', type=list, default=[30, 30, 30])
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize (default: 128)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N', help='test batchsize (default: 100)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')

parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--data', type=str, default='/mnt/lustre/luozhipeng/data/ImageNet') # need modified

parser.add_argument('--save_root', type=str, default=save_base_root_check)
parser.add_argument('--TIMES', type=int, default=2)
parser.add_argument('--Begin_TIMES', type=int, default=1)


def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)
    if args.cuda:
        cudnn.benchmark = True

    net = vgg19(pretrained=False)
    if args.cuda:
        net = torch.nn.DataParallel(net).cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()
    criterions = {'criterionCls': criterionCls}

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    pre_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            pre_normalize, ])), batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            pre_normalize, ])), batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    for TIMES in range(args.Begin_TIMES, args.TIMES):
        print('Training for the {}-th time! '.format(TIMES))
        global save_base_root_times
        save_base_root_times = args.save_root + '/{}'.format(TIMES)
        if not os.path.exists(save_base_root_times):
            os.mkdir(save_base_root_times)
        global save_max_accu
        save_max_accu = args.save_root + '/save_max_log.txt'

        save_loss = save_base_root_times + '/{}.txt'.format(python_name)
        max_prec = [0.0, 0, 0.0, 0]
        batch_time_list = [0.0, 100.0]
        for epoch in range(1, args.epochs + 1):
            pre11, pre15, batch_time = one_epoch(net, train_loader, test_loader, criterions, optimizer, epoch, save_loss, TIMES)
            max, batch = compare_max(max_prec, pre11, pre15, batch_time_list, batch_time, epoch, net, save_max_accu)
            max_prec = max
            batch_time_list = batch


def compare_max(max_prec, pre11, pre15, batch_time_list, batch_time, epoch, net, save_max_log):
    if max_prec[0] < pre11:
        max_prec[0] = pre11
        max_prec[1] = epoch
        save_dir1 = save_base_root_times + '/Max-'
        if not os.path.exists(save_dir1):
            os.mkdir(save_dir1)
        save_name1 = '{}_{}.ckp'.format(args.name, epoch)
        save_name1 = os.path.join(save_dir1, save_name1)
        ckp1 = os.listdir(save_dir1)
        for i in ckp1:
            ckp_path1 = os.path.join(save_dir1, i)
            os.remove(ckp_path1)
        save_checkpoint({'epoch': epoch, 'net': net.state_dict(), }, save_name1)
    elif max_prec[2] < pre15:
        max_prec[2] = pre15
        max_prec[3] = epoch
    else:
        pass

    if epoch % 10 == 0:
        save_dir2 = save_base_root_times + '/Every10_ckp'
        if not os.path.exists(save_dir2):
            os.mkdir(save_dir2)
        save_name2 = '{}_{}.ckp'.format(args.name, epoch)
        save_name2 = os.path.join(save_dir2, save_name2)
        ckp2 = os.listdir(save_dir2)
        for i in ckp2:
            ckp_path2 = os.path.join(save_dir2, i)
            os.remove(ckp_path2)
        save_checkpoint({'epoch': epoch, 'net': net.state_dict(), }, save_name2)

    if batch_time_list[0] < batch_time:
        batch_time_list[0] = batch_time
    elif batch_time_list[1] > batch_time:
        batch_time_list[1] = batch_time
    else:
        pass
    current_max = 'Net:{} || Current top11: [{:.2f}/{}], top15: [{:.2f}/{}], batch_time: [{:.4f}/{:.4f}]' \
        .format(python_name, max_prec[0], max_prec[1], max_prec[2], max_prec[3], batch_time_list[0], batch_time_list[1])
    print(current_max)
    if epoch == args.epochs:
        with open(save_max_log, 'a') as f:
            f.write(str(current_max))
            f.write('\n')
            f.close()
    return max_prec, batch_time_list


def one_epoch(net, train_loader, test_loader, criterions, optimizer, epoch, save_loss, times):
    epoch_start_time = time.time()
    adjust_lr(optimizer, epoch, times)
    train(train_loader, net, optimizer, criterions, epoch)
    epoch_time = time.time() - epoch_start_time
    print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

    print('testing the models......')
    test_start_time = time.time()
    pre11, pre15, batch_time = test(test_loader, net, epoch, criterions, save_loss)
    test_time = time.time() - test_start_time
    print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))
    return pre11, pre15, batch_time


def train(train_loader, net, optimizer, criterions, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    top11 = AverageMeter()
    top15 = AverageMeter()

    net.train()
    criterionCls = criterions['criterionCls']

    end = time.time()
    for idx, (img, target) in enumerate(train_loader, start=1):  # idx increase from 1
        data_time.update(time.time() - end)
        img = cpu_gpu(args.cuda, img, volatile=False)
        target = cpu_gpu(args.cuda, target, volatile=False)
        out = net(img)

        cls = criterionCls(out, target)  # CrossEntropy
        prec11, prec15 = accuracy(out, target, topk=(1, 5))
        cls_losses.update(cls.item(), img.size(0))
        top11.update(prec11.item(), img.size(0))
        top15.update(prec15.item(), img.size(0))

        optimizer.zero_grad()
        cls.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}], '
                  'Time:{batch_time.val:.4f}, '
                  'Data:{data_time.val:.4f}, '
                  'cls-loss:[{cls_losses.avg:.2f}], ' 
                  'prec@1_1:{top11.val:.2f}({top11.avg:.2f}), '
                  'prec@1_5:{top15.val:.2f}({top15.avg:.2f})'
                  .format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                          cls_losses=cls_losses, top11=top11, top15=top15))


def test(test_loader, net, epoch, criterions, save_loss):
    cls_losses = AverageMeter()
    top11 = AverageMeter()
    top15 = AverageMeter()
    batch_time = AverageMeter()

    criterionCls = criterions['criterionCls']
    net.eval()

    end = time.time()
    for idx, (img, target) in enumerate(test_loader, start=1):
        img = cpu_gpu(args.cuda, img, volatile=True)
        target = cpu_gpu(args.cuda, target, volatile=True)
        out = net(img)

        # for T
        cls1 = criterionCls(out, target)  # CrossEntropy

        prec11, prec15 = accuracy(out, target, topk=(1, 5))
        cls_losses.update(cls1.item(), img.size(0))
        top11.update(prec11.item(), img.size(0))
        top15.update(prec15.item(), img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    result = 'Epoch[{0}], Time:{batch_time.val:.4f}, ' \
             'cls-loss:[{cls_losses.avg:.2f}], ' \
             'prec@1_1:{top11.val:.2f}({top11.avg:.2f}), prec@1_5:{top15.val:.2f}({top15.avg:.2f})'.\
        format(epoch, batch_time=batch_time, cls_losses=cls_losses, top11=top11, top15=top15)
    print(result)

    write_result = 'Epoch[{0}], Time:{batch_time.val:.4f}, ' \
                   'prec@1_1:{top11.val:.2f}({top11.avg:.2f}), ' \
                   'prec@1_5:{top15.val:.2f}({top15.avg:.2f})'.format(
                    epoch, batch_time=batch_time, top11=top11, top15=top15)
    with open(save_loss, 'a') as f:
        f.write(write_result)
        f.write('\n')
        f.close()
    return top11.avg, top15.avg, batch_time.avg


import math
def adjust_lr(optimizer, epoch, times):
    scale = 0.1
    lr_list = []
    for i in range(len(args.epoch_list)):
        lr_list += [args.lr * math.pow(scale, i)] * args.epoch_list[i]
    lr = lr_list[epoch - 1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Times:{} || Epoch: [{}], lr: {}'.format(times, epoch, lr))


if __name__ == '__main__':
    main()