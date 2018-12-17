#train
import os
import time
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import MobileNetV2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

##
#
#
#

import  numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main():
   
    np.random.seed(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    random.seed(666)

   
    file_name = os.path.basename(__file__).split('.')[0]
    
    if not os.path.exists('./model/%s' % file_name):
        os.makedirs('./model/%s' % file_name)
    if not os.path.exists('./result/%s' % file_name):
        os.makedirs('./result/%s' % file_name)
 
    if not os.path.exists('./result/%s.txt' % file_name):
        with open('./result/%s.txt' % file_name, 'w') as acc_file:
            pass
    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), file_name))

   
    def default_loader(path):
       
        return Image.open(path).convert('RGB')


    class TrainDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)

    class ValDataset(Dataset):
        def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
            imgs = []
            for index, row in label_list.iterrows():
                imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

        def __getitem__(self, index):
            filename, label = self.imgs[index]
            img = self.loader(filename)
            if self.transform is not None:
                img = self.transform(img)
            return img, label

        def __len__(self):
            return len(self.imgs)


    
    class FixedRotation(object):
        def __init__(self, angles):
            self.angles = angles

        def __call__(self, img):
            return fixed_rotate(img, self.angles)

    def fixed_rotate(img, angles):
        angles = list(angles)
        angles_num = len(angles)
        index = random.randint(0, angles_num - 1)
        return img.rotate(angles[index])

   
    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        
        train_diff_loss =[]
        train_diff_acc =[]
        model.train()

        end = time.time()
      
        for i, (images, target) in enumerate(train_loader):
          
            data_time.update(time.time() - end)
  
            image_var = torch.tensor(images).cuda(async=True)
            label = torch.tensor(target).cuda(async=True)

          
            y_pred = model(image_var)
          
            loss = criterion(y_pred, label)
            losses.update(loss.item(), images.size(0))

          
            prec, PRED_COUNT = accuracy(y_pred.data, target, topk=(1, 1))
            acc.update(prec, PRED_COUNT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))
                train_diff_loss.append(losses.val)
                train_diff_acc.append(acc.val)
        return train_diff_acc,train_diff_loss

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        model.eval()

        end = time.time()
        
        c_m =[]
        for i in range(8):
            c_m.append([])
            for j in range(8):
                c_m[i].append(0)
        
        
        for i, (images, labels) in enumerate(val_loader):
            image_var = torch.tensor(images).cuda(async=True)
            target = torch.tensor(labels).cuda(async=True)

            with torch.no_grad():
                y_pred = model(image_var)
                loss = criterion(y_pred, target)

          
            prec, PRED_COUNT= accuracy(y_pred.data, labels, topk=(1, 1))
            c_m = confusion_matrix(y_pred.data,labels,c_m,topk=(1,1))
            losses.update(loss.item(), images.size(0))
            acc.update(prec, PRED_COUNT)

           
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('TrainVal: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc))
              
        print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
              ' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
        return acc.avg, losses.avg ,c_m

   
    def save_checkpoint(state, is_best, is_lowest_loss, filename='./model/%s/checkpoint.pth.tar' % file_name):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './model/%s/model_best.pth.tar' % file_name)
        if is_lowest_loss:
            shutil.copyfile(filename, './model/%s/lowest_loss.pth.tar' % file_name)

   
    class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

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
    def averagenum(num):
        nsum = 0
        for i in range(len(num)):
            nsum += num[i]
        return nsum / len(num)

    def plotCM(classes, matrix, savname):
        # Normalize by row
        matrix = matrix.astype(np.float)
        print(matrix)
        linesum = matrix.sum(1)
        linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
        matrix /= linesum
        # plot
        plt.switch_backend('agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix)
        fig.colorbar(cax)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        for i in range(matrix.shape[0]):
            ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
        ax.set_xticklabels([''] + classes, rotation=90)
        ax.set_yticklabels([''] + classes)
        #save
        plt.savefig(savname)
    def adjust_learning_rate():
        nonlocal lr
        lr = lr / lr_decay
        return optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
    def plot_train_loss_acc(loss,acc):
        x1 = range(0,70)
        x2 = range(0,70)
        y1 = acc
        y2 = loss
        plt.subplot(2,1,1)
        plt.plot(x1,y1,'o-')
        plt.title('Train acc vs. epoches')
        plt.ylabel('Train acc')
        plt.subplot(2,1,2)
        plt.plot(x2,y2,'.-')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        plt.show()
        plt.savefig('/home/luxin/tianchi_lvcai-master/gesture_task/train_acc_loss.jpg')
    def plot_val_loss_acc(loss,acc):
        x1 = range(0,70)
        x2 = range(0,70)
        y1 = acc
        y2 = loss
        plt.subplot(2,1,1)
        plt.plot(x1,y1,'o-')
        plt.title('val acc vs. epoches')
        plt.ylabel('val acc')
        plt.subplot(2,1,2)
        plt.plot(x2,y2,'.-')
        plt.xlabel('val loss vs. epoches')
        plt.ylabel('val loss')
        plt.show()
        plt.savefig('/home/luxin/tianchi_lvcai-master/gesture_task/val_acc_loss.jpg')

    def count_num_max(list_count):
        my_set = set(list_count)
        my_dict ={}
        for item in my_set:
            my_dict.update({item:list_count.count(item)})
        tmp =0 
        result_class = 0
        for item in my_dict:
            if tmp <= my_dict[item]:
                tmp = my_dict[item]
                result_class = item
        return result_class


    def accuracy(y_pred, y_actual, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
        PRED_CORRECT_COUNT = 0
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)
        #######3#3###33######
        
        for j in range(pred.size(0)):
            if int(y_actual[j]) == int(pred[j]):
                PRED_CORRECT_COUNT += 1
        if PRED_COUNT == 0:
            final_acc = 0
        else:
            final_acc = PRED_CORRECT_COUNT / PRED_COUNT
        return final_acc * 100, PRED_COUNT

#
#
#
    def confusion_matrix(y_pred, y_actual,c_m, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        final_acc = 0
        maxk = max(topk)
        # for prob_threshold in np.arange(0, 1, 0.01):
        PRED_COUNT = y_actual.size(0)
       
        prob, pred = y_pred.topk(maxk, 1, True, True)
        # prob = np.where(prob > prob_threshold, prob, 0)
        #######3#3###33######
#        c_m =[]
#        for i in range(8):
#            c_m.append([])
#            for j in range(8):
#                c_m[i].append(0)
#        print(pred.size(0))
        for i in range(pred.size(0)):
            c_m[int(y_actual[i])][int(pred[i])]+=1
#        print(c_m)
        return c_m


    
    os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5'
   
    batch_size = 37
    
    workers = 12

    
    stage_epochs = [20, 20, 20, 10]  
   
    lr = 0.001
    
    lr_decay = 5
   
    weight_decay = 1e-4

   
    stage = 0
    start_epoch = 0
    total_epochs = sum(stage_epochs)
    best_precision = 0
    lowest_loss = 100

    print_freq = 1
   
    val_ratio = 0.12
  
    evaluate = False
  
    resume = False
    
    model = MobileNetV2.v2(num_classes=8)
    model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if resume:
        checkpoint_path = './model/%s/checkpoint.pth.tar' % file_name
        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            best_precision = checkpoint['best_precision']
            lowest_loss = checkpoint['lowest_loss']
            stage = checkpoint['stage']
            lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            
            if start_epoch in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

  
    all_data = pd.read_csv('data/label.csv')
   
    train_data_list, val_data_list = train_test_split(all_data, test_size=val_ratio, random_state=666, stratify=all_data['label'])
   

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
   
    train_data = TrainDataset(train_data_list,
                              transform=transforms.Compose([
                                  transforms.Resize((352, 352)),
                                  transforms.ColorJitter(0.15, 0.15, 0.15, 0.075),
                                 # transforms.RandomHorizontalFlip(),
                                  transforms.RandomGrayscale(),
                                  # transforms.RandomRotation(20),
                                 # FixedRotation([0, 90, 120,180, 270]),
                                  transforms.RandomCrop(320),
                                  transforms.ToTensor(),
                                  normalize,
                              ]))

  
    val_data = ValDataset(val_data_list,
                          transform=transforms.Compose([
                              transforms.Resize((352, 352)),
                              transforms.CenterCrop(320),
                              transforms.ToTensor(),
                              normalize,
                          ]))


    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_loader = DataLoader(val_data, batch_size=batch_size*2, shuffle=False, pin_memory=False, num_workers=workers)
   
    criterion = nn.CrossEntropyLoss().cuda()

   
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)

    if evaluate:
        validate(val_loader, model, criterion)
    else:
        all_train_diff_acc =[]
        all_train_diff_loss =[]
        all_val_diff_acc =[]
        all_val_diff_loss =[]
        classes=['none','up','down','left','right','forward','back','stop']
        for epoch in range(start_epoch, total_epochs):
            # train for one epoch
            train_diff_acc, train_diff_loss =train(train_loader, model, criterion, optimizer, epoch)
            
            all_train_diff_acc.append(averagenum(train_diff_acc))
            all_train_diff_loss.append(averagenum(train_diff_loss))
            # evaluate on validation set
            precision, avg_loss,CM= validate(val_loader, model, criterion)
            all_val_diff_acc.append(precision)
            all_val_diff_loss.append(avg_loss)
            plotCM(classes,np.array(CM),'tmp'+'%d'%(epoch))
            
            
            with open('./result/%s.txt' % file_name, 'a') as acc_file:
                acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
 
     
            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_precision': best_precision,
                'lowest_loss': lowest_loss,
                'stage': stage,
                'lr': lr,
            }
            save_checkpoint(state, is_best, is_lowest_loss)
 
            
            if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
                stage += 1
                optimizer = adjust_learning_rate()
                model.load_state_dict(torch.load('./model/%s/model_best.pth.tar' % file_name)['state_dict'])
                print('Step into next stage')
                with open('./result/%s.txt' % file_name, 'a') as acc_file:
                    acc_file.write('---------------Step into next stage----------------\n')
 
    plot_train_loss_acc(all_train_diff_loss,all_train_diff_acc)
    plot_val_loss_acc(all_val_diff_loss,all_val_diff_acc)
    with open('./result/%s.txt' % file_name, 'a') as acc_file:
        acc_file.write('* best acc: %.8f  %s\n' % (best_precision, os.path.basename(__file__)))
    with open('./result/best_acc.txt', 'a') as acc_file:
        acc_file.write('%s  * best acc: %.8f  %s\n' % (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision, os.path.basename(__file__)))
 
    torch.cuda.empty_cache()
 
if __name__ == '__main__':
    main()
