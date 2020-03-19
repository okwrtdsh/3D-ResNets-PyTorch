import os
os.environ['PYTHONHASHSEED'] = '0'
import random
random.seed(12345)
import numpy as np
np.random.seed(42)

import torch
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
torch.backends.cudnn.enabled = True

from torch.backends import cudnn
cudnn.benchmark = True

import json
import numpy as np # noqa
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model_decoder import generate_model
from mean import get_mean, get_std
from decoder_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, RGB2Gray, LowResolution)
from temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from spatio_temporal_transforms import Coded, Averaged, OneFrame, ToTemporal, ToRepeat
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompos # noqa
from dataset_decoder import get_training_set, get_validation_set, get_test_set
from utils import Logger
# from train_decoder import train_epoch
# from validation_decoder import val_epoch
import test
import re




###########################################################
import sys
import math
import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np

from utils import AverageMeter, calculate_accuracy, save_gif, accuracy
from models.binarized_modules import binarizef


def train_epoch(epoch, data_loader, model, criterion_decoder, criterion_clf, optimizer, opt,
                epoch_logger, batch_logger, device):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_mse = AverageMeter()
    loss_ce = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, target_labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs.to(device)
        targets = targets.to(device)
        target_labels = target_labels.to(device)
        outputs, outputs_clf = model(inputs)
        loss1 = criterion_decoder(outputs, targets)
        loss2 = criterion_clf(outputs_clf, target_labels)
        loss = loss1 + loss2 * opt.alpha
        prec1, prec5 = accuracy(outputs_clf.data, target_labels, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.item(), inputs.size(0))
        loss_mse.update(loss1.item(), inputs.size(0))
        loss_ce.update(loss2.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        sys.stdout.flush()
        sys.stdout.write('\rEpoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'MSE: {mse.val:.4f} ({mse.avg:.4f})\t'
                         'PSNR: {psnr_val:.4f} ({psnr_avg:.4f})\t'
                         'CE: {ce.val:.4f} ({ce.avg:.4f})\t'
                         'Acc@1: {top1.val:.4f} ({top1.avg:.4f})\t'
                         'Acc@5: {top5.val:.4f} ({top5.avg:.4f})\t'
                         '\t\t'.format(
                             epoch,
                             i + 1,
                             len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             mse=loss_mse,
                             ce=loss_ce,
                             top1=top1,
                             top5=top5,
                             psnr_val=10 * math.log10(1 / loss_mse.val),
                             psnr_avg=10 * math.log10(1 / loss_mse.avg),
                         ))
    sys.stdout.flush()
    print('\n[Train] Epoch{0}\t'
          'Time: {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
          'Data: {data_time.sum:.3f} ({data_time.avg:.3f})\t'
          'Loss: {loss.avg:.4f}\n'
          'MSE: {mse.avg:.4f}\t'
          'PSNR: {psnr_avg:.4f}\t'
          'CE: {ce.avg:.4f}\t'
          'Acc@1: {top1.avg:.4f}\t'
          'Acc@5: {top5.avg:.4f}\t'
          '\t\t'.format(
              epoch,
              batch_time=batch_time,
              data_time=data_time,
              loss=losses,
              mse=loss_mse,
              ce=loss_ce,
              top1=top1,
              top5=top5,
              psnr_avg=10 * math.log10(1 / loss_mse.avg),
          ))
    print()

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'mse': loss_mse.avg,
        'psnr': 10 * math.log10(1 / loss_mse.avg),
        'ce': loss_ce.avg,
        'top1': top1.avg,
        'top5': top5.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'batch_time': batch_time.sum,
        'data_time': data_time.sum,
    })
    if 'exp' in opt.model and not opt.load_path:
        mask = binarizef(
            list(model.parameters())[0]
        ).add_(1).div_(2).to('cpu').detach().numpy()
        print('max', mask.max())
        print('min', mask.min())
        mask = mask.reshape((opt.sample_duration, 8, 8, 1)).astype(np.uint8)
        assert mask.shape == (opt.sample_duration, 8, 8, 1)
        # save_file_path = os.path.join(opt.result_path,
        #                       'mask_{}.npy'.format(epoch))
        # np.save(save_file_path, mask)
        save_file_path = os.path.join(opt.result_path,
                                      'mask_{}.gif'.format(epoch))
        save_gif(mask, save_file_path, vmax=1, vmin=0)

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


###########################################################

import torch
from torch.autograd import Variable
import time
import sys
import numpy as np

from utils import AverageMeter, calculate_accuracy, accuracy


def val_epoch(epoch, data_loader, model, criterion_decoder, criterion_clf, opt, logger, device):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_mse = AverageMeter()
    loss_ce = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, target_labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs.to(device)
        targets = targets.to(device)
        target_labels = target_labels.to(device)
        outputs, outputs_clf = model(inputs)
        loss1 = criterion_decoder(outputs, targets)
        loss2 = criterion_clf(outputs_clf, target_labels)
        loss = loss1 + loss2 * opt.alpha

        losses.update(loss.item(), inputs.size(0))
        loss_mse.update(loss1.item(), inputs.size(0))
        loss_ce.update(loss2.item(), inputs.size(0))

        prec1, prec5 = accuracy(outputs_clf.data, target_labels, topk=(1, 5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))


        if i == 0:
            outputs = outputs.cpu().detach().numpy().reshape(-1, 16, 112, 112, 1)
            for j, output in enumerate(outputs):
                if j % 3 == 0 and j < 10:
                    save_gif_path = os.path.join(opt.result_path, 'val_%005d_sample%02d.gif' % (epoch, j))
                    save_gif((np.clip(output, 0, 1) * 255).reshape(16, 112, 112, 1).astype(np.uint8), save_gif_path, vmax=255, vmin=0, interval=2000/16)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        sys.stdout.flush()
        sys.stdout.write('\rEpoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'MSE: {mse.val:.4f} ({mse.avg:.4f})\t'
                         'PSNR: {psnr_val:.4f} ({psnr_avg:.4f})\t'
                         'CE: {ce.val:.4f} ({ce.avg:.4f})\t'
                         'Acc@1: {top1.val:.4f} ({top1.avg:.4f})\t'
                         'Acc@5: {top5.val:.4f} ({top5.avg:.4f})\t'
                         '\t\t'.format(
                             epoch,
                             i + 1,
                             len(data_loader),
                             batch_time=batch_time,
                             data_time=data_time,
                             loss=losses,
                             mse=loss_mse,
                             ce=loss_ce,
                             top1=top1,
                             top5=top5,
                             psnr_val=10 * math.log10(1 / loss_mse.val),
                             psnr_avg=10 * math.log10(1 / loss_mse.avg),
                         ))
    sys.stdout.flush()
    print('\n[Val] Epoch{0}\t'
          'Time: {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
          'Data: {data_time.sum:.3f} ({data_time.avg:.3f})\t'
          'Loss: {loss.avg:.4f}\n'
          'MSE: {mse.avg:.4f}\t'
          'PSNR: {psnr_avg:.4f}\t'
          'CE: {ce.avg:.4f}\t'
          'Acc@1: {top1.avg:.4f}\t'
          'Acc@5: {top5.avg:.4f}\t'
          '\t\t'.format(
              epoch,
              batch_time=batch_time,
              data_time=data_time,
              loss=losses,
              mse=loss_mse,
              ce=loss_ce,
              top1=top1,
              top5=top5,
              psnr_avg=10 * math.log10(1 / loss_mse.avg),
          ))
    print()
    logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'mse': loss_mse.avg,
        'psnr': 10 * math.log10(1 / loss_mse.avg),
        'ce': loss_ce.avg,
        'top1': top1.avg,
        'top5': top5.avg,
    })
    return losses.avg

###########################################################
import numpy as np
from glob import glob
import cv2
import math
from utils import save_gif


def psnr(img1, img2, vmax=1):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10(vmax * vmax / mse)


def eval_epoch(epoch, model, opt, device):
    model.eval()
    psnrs = []
    size = opt.spatial_compress_size
    GT = []
    DATA = []
    paths = sorted(glob('../EVAL_mat/EVAL14/*.npy'))
    for i, path in enumerate(paths):
        video = np.load(path).reshape(16, 256, 256)
        video = np.array([cv2.resize(img, dsize=(112, 112), fx=1/size, fy=1/size).astype(np.uint8) for img in video])
        GT.append(video.astype(np.float32) / 255)
        DATA.append(np.array([cv2.resize(img, dsize=None, fx=1/size, fy=1/size).astype(np.uint8) for img in video]).astype(np.float32) / 255)
    GT = np.array(GT).astype(np.float32)
    DATA = np.array(DATA).astype(np.float32)
    reconstructed = []
    with torch.no_grad():
        for i, path in enumerate(paths):
            data = torch.from_numpy(DATA[i].reshape(1, 1, 16, 112//size, 112//size)).to(device).float()
            output, _ = model(data)
            output = output.cpu().detach().numpy().reshape(1, 16, 112, 112)
            reconstructed.append(output)

    for i, path in enumerate(paths):
        p = psnr(GT[i],  np.clip(reconstructed[i], 0, 1), vmax=1)
        print(os.path.basename(path).replace('.npy', ':'), p)
        save_gif_path = os.path.join(opt.result_path, ('eval_%005d_' % epoch) + os.path.basename(path).replace('.npy', '.gif'))
        save_gif((np.clip(reconstructed[i], 0, 1) * 255).reshape(16, 112, 112, 1).astype(np.uint8), save_gif_path, vmax=255, vmin=0, interval=2000/16)
        psnrs.append(p)
    print(np.mean(psnrs))
###########################################################




if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        #opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file, indent=2)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    from torch.backends import cudnn
    cudnn.benchmark = True
    criterion_decoder = nn.MSELoss()
    criterion_clf = nn.CrossEntropyLoss()
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    if not opt.no_cuda:
        criterion_decoder = criterion_decoder.to(device)
        criterion_clf = criterion_clf.to(device)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        common_temporal_transform = TemporalRandomCrop(opt.sample_duration)
        common_spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            RGB2Gray(),
        ])
        target_spatial_transform = Compose([
            ToTensor(opt.norm_value), norm_method,
        ])
        input_spatial_transform = Compose([
            LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
            ToTensor(opt.norm_value), norm_method,
        ])
        target_label_transform = ClassLabel()
        training_data = get_training_set(
            opt, common_temporal_transform, common_spatial_transform,
            target_spatial_transform, input_spatial_transform, target_label_transform)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'mse', 'psnr', 'ce', 'top1', 'top5', 'lr', 'batch_time', 'data_time'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'lr'])

        sv_rgx = re.compile(r'.*\.weight\d+\..*')
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.optimizer == 'sgd':
            params = []
            for i, ((name, _), p) in enumerate(zip(model.named_parameters(), parameters)):
                if 'exp' in name:
                    print('{}*: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * opt.lr_exp_rate})
                elif sv_rgx.match(name):
                    print('{}**: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                else:
                    print('{}: {}({})'.format(i, name, p.shape))
                    params.append({"params": p})
            optimizer = optim.SGD(
                params,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        elif opt.optimizer == 'adam':
            params = []
            for i, ((name, _), p) in enumerate(zip(model.named_parameters(), parameters)):
                if 'exp' in name:
                    print('{}*: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * opt.lr_exp_rate})
                elif sv_rgx.match(name):
                    print('{}**: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                else:
                    print('{}: {}({})'.format(i, name, p.shape))
                    params.append({"params": p})
            optimizer = optim.Adam(
                params,
                lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            params = []
            for i, ((name, _), p) in enumerate(zip(model.named_parameters(), parameters)):
                if 'exp' in name:
                    print('{}*: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * opt.lr_exp_rate})
                elif sv_rgx.match(name):
                    print('{}**: {}({})'.format(i, name, p.shape))
                    params.append({
                        "params": p,
                        "lr": opt.learning_rate * 64 * opt.lr_pt_rate})
                else:
                    print('{}: {}({})'.format(i, name, p.shape))
                    params.append({"params": p})
            optimizer = optim.RMSprop(
                params,
                lr=opt.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        common_temporal_transform = TemporalCenterCrop(opt.sample_duration)
        common_spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            RGB2Gray(),
        ])
        target_spatial_transform = Compose([
            ToTensor(opt.norm_value), norm_method,
        ])
        input_spatial_transform = Compose([
            LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
            ToTensor(opt.norm_value), norm_method,
        ])
        target_label_transform = ClassLabel()

        validation_data = get_validation_set(
            opt, common_temporal_transform, common_spatial_transform,
            target_spatial_transform, input_spatial_transform, target_label_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'),
            ['epoch', 'loss', 'mse', 'psnr', 'ce', 'top1', 'top5'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.fixed_mask:
        check = False
        for i, (name, param) in enumerate(model.named_parameters()):
            if name == 'module.exp.weight':
                print(name, 'FIXED!')
                param.requires_grad = False
                check = True
        if not check:
            raise

    model.to(device)
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion_decoder, criterion_clf, optimizer, opt,
                        train_logger, train_batch_logger, device)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion_decoder, criterion_clf, opt,
                                        val_logger, device)
        if i % 5 == 0:
            eval_epoch(i, model, opt, device)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        common_temporal_transform = LoopPadding(opt.sample_duration)
        common_spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            RGB2Gray(),
        ])
        target_spatial_transform = Compose([
            ToTensor(opt.norm_value), norm_method,
        ])
        input_spatial_transform = Compose([
            LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
            ToTensor(opt.norm_value), norm_method,
        ])
        target_label_transform = VideoID()

        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            RGB2Gray(),
            ToTensor(opt.norm_value), norm_method,
        ])

        test_data = get_test_set(
            opt, common_temporal_transform, common_spatial_transform,
            target_spatial_transform, input_spatial_transform, target_label_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names, device)



