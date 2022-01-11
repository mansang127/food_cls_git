import numpy as np
import shutil
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config as cfg


class AverageMeter(object):           
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset() #一开始就初始化下面4个值

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def logger(info, file_path=cfg.log_path, flag=True, init=False):
    
    if init:
        with open(file_path, 'w') as fo:
            pass
        return
    
    if flag:
        print(info)
    with open(file_path, 'a') as fo:
        fo.write(info + '\n')
        
    return

    
def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified values of k """
    #返回两个值：只考虑概率最大的准确率、同时考虑概率前五的准确率（只要有一个对就算对）
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0) #target.shape=torch.Size([128])

        #k：指明是得到前k个数据以及其index
        #dim
        #largest：如果为True，按照大到小排序； 如果为False，按照小到大排序
        #sorted：返回的结果按照顺序返回
        _, pred = output.topk(maxk, 1, True, True) #pred为索引值
        pred = pred.t() #转置，此时pred维度为5x128，第一行为最大值对应的索引
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #target由1x128扩充为5x128

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size)) #100是百分比准确率，batchsize是求平均
        return res
     
     
def adjust_learning_rate(optimizer, epoch, cfg):
    """ Sets the learning rate """
    if cfg.cos:
        lr_min = 0
        lr_max = cfg.lr
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / cfg.num_epochs * 3.1415926535))
    else:
        epoch = epoch + 1
        if epoch <= 5:
            lr = cfg.lr * epoch / 5
        elif epoch > 80:
            lr = cfg.lr * 0.01
        elif epoch > 60:
            lr = cfg.lr * 0.1
        else:
            lr = cfg.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, model_dir):
    ''' save ckeck point current and the best '''
    filename = model_dir + 'ckpt/current.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_dir + 'ckpt/model_best.pth.tar')
     
        
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')
    
    model.train()
    training_data_num = len(train_loader.dataset)
    end_steps = int(training_data_num / train_loader.batch_size)
    end = time.time()
    
    for i, (images, target) in enumerate(tqdm(train_loader)):
        if i > end_steps:
            break

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        acc1 = acc1[0].detach().cpu().numpy()
        acc5 = acc5[0].detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        logger('*[Iter]: {:03d} [Acc@1]: {:.3f}%  [Acc@5]: {:.3f}%  [Loss]: {:.5f}.'.format(i, acc1, acc5, loss), flag=False)
        
    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f') #name和format
    losses = AverageMeter('Loss', ':.3f')
    top1 = AverageMeter('Acc@1', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.3f')

    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(cfg.num_classes).cuda()
    correct = torch.zeros(cfg.num_classes).cuda()

    confidence = np.array([])
    pred_class = np.array([])
    true_class = np.array([])

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(cfg.gpu, non_blocking=True)


            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0)) #累计当前loss
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            _, predicted = output.max(1) #ouput:128x101, max(1)返回行最大
            target_one_hot = F.one_hot(target, cfg.num_classes) #128x101
            predict_one_hot = F.one_hot(predicted, cfg.num_classes) #128x101
            class_num = class_num + target_one_hot.sum(dim=0).to(torch.float) #列求和进行种类统计
            correct = correct + (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float) #101个种类的正确数量统计

            prob = torch.softmax(output, dim=1) #维度不变，每行和为1
            confidence_part, pred_class_part = torch.max(prob, dim=1) #value, index
            confidence = np.append(confidence, confidence_part.cpu().numpy()) #每个循环+128个元素，值为概率（数据要返回cpu变为ndarray类）
            pred_class = np.append(pred_class, pred_class_part.cpu().numpy()) #每个循环+128个元素，值为索引（即预测种类）
            true_class = np.append(true_class, target.cpu().numpy()) #每个循环+128个元素，值为真实类别
            
            batch_time.update(time.time() - end)
            end = time.time()

        logger('* Acc@1 {top1.avg:.3f}% Acc@5 {top5.avg:.3f}%.'.format(top1=top1, top5=top5))

    return top1.avg

