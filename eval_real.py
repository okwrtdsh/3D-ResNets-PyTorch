import os
import numpy as np # noqa
import matplotlib as mpl
mpl.use('svg')  # noqa

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
import json
import pickle
import torch
from torch import nn
# from torch import optim
# from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor, RGB2Gray, LowResolution)
from temporal_transforms import LoopPadding, TemporalCenterCrop
from spatio_temporal_transforms import Coded, Averaged, OneFrame, ToTemporal  # , ToRepeat
from target_transforms import ClassLabel  # , VideoID
from target_transforms import Compose as TargetCompos # noqa
from dataset import get_test_set
from utils import Logger
from utils import AverageMeter, accuracy
from glob import glob


def test_eval(data_loader, model, criterion, opt, logger, device):
    print('eval')

    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    inst_results = []
    inst_targets = []
    input_mean = []
    N = 100
    for i, (inputs, targets) in enumerate(data_loader):
        samples = inputs.to('cpu').detach().numpy()
        for j in range(opt.batch_size):
            n = i * opt.batch_size + j
            if n > N:
                break
            if j == len(samples):
                break
            if opt.compress in ["one", "avg"]:
                from PIL import Image
                save_file_path = os.path.join(opt.result_path, 'sample_%05d.png' % n)
                Image.fromarray(samples[j].reshape((112, 112)).astype(np.uint8)).resize((1120, 1120)).save(save_file_path)
            elif opt.compress in ["mask"]:
                from PIL import Image
                save_file_path = os.path.join(opt.result_path, 'sample_%05d.png' % n)
                Image.fromarray(np.clip(samples[j] * 3, 0, 255).reshape((112, 112)).astype(np.uint8)).resize((1120, 1120)).save(save_file_path)
            else:
                from utils import save_gif
                save_gif_path = os.path.join(opt.result_path, 'sample_%005d.gif' % n)
                save_gif(samples[j].reshape(16, 112, 112, 1).astype(np.uint8), save_gif_path, vmax=255, vmin=0, interval=2000/16)

        input_mean.extend([i.mean() for i in inputs.detach().cpu().numpy()])
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)

        inst_results.append(outputs.detach().cpu().numpy())
        inst_targets.append(targets.detach().cpu().numpy())

        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))

        prec1, prec3, prec5 = accuracy(outputs.data, targets, topk=(1, 3, 5))
        top1.update(prec1, inputs.size(0))
        top3.update(prec3, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        sys.stdout.flush()
        sys.stdout.write('\rEVAL: [{:>6}/{:>6}] '
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Acc@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                         'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t\t'
                         'len {len_mean},'
                         'mean {mean:.4f},'
                         'std {std:.4f},'
                         'min {min:.4f},'
                         'max {max:.4f}'
                         '\t\t'.format(
                             i + 1,
                             len(data_loader),
                             loss=losses,
                             top1=top1,
                             top3=top3,
                             top5=top5,
                             len_mean=len(input_mean),
                             mean=np.mean(input_mean),
                             std=np.std(input_mean),
                             min=np.min(input_mean),
                             max=np.max(input_mean),
                         ))
    print()
    logger.log({
        'top1': top1.avg,
        'top3': top3.avg,
        'top5': top5.avg,
    })

    res = {
        'inst_results': inst_results,
        'inst_targets': inst_targets,
        'losses': losses,
        'top1': top1,
        'top3': top3,
        'top5': top5,
    }
    with open(os.path.join(opt.result_path, 'res.pkl'), 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(opt.result_path, 'res.pkl'), 'rb') as f:
    #     res = pickle.load(f)

    # inst_results = res['inst_results']
    # inst_targets = res['inst_targets']
    # losses = res['losses']
    # top1 = res['top1']
    # top3 = res['top3']
    # top5 = res['top5']

    inst_targets = np.concatenate(inst_targets, axis=0)
    print(inst_targets.shape)
    inst_results = np.concatenate(inst_results, axis=0)
    print(inst_results.shape)

    with open(opt.annotation_path, 'r') as f:
        label_to_id = json.load(f)
    labels = list(label_to_id.keys())

    res = []
    res3 = []
    res5 = []
    if os.path.basename(opt.video_path) in ["mask", "mask2", "mask3", "mask_old"]:
        npys = sorted(glob(os.path.join(opt.video_path, "*/clip[0-9]/npy/*.npy")),key=lambda x: x.split('/'))
        y_pred = inst_results.argmax(axis=1)
        y_true = inst_targets
        print(y_pred)
        print(y_true)

        prev_cid = None
        d = None
        d3 = None
        for i, cid in enumerate(y_true):
            if prev_cid != cid:
                prev_cid = cid
                if d is not None:
                    for k,v in d.items():
                        print(k, np.max(v), np.mean(v), np.array(v).argmax())
                        res.append(np.max(v))
                    for k,v in d3.items():
                        print(k, np.max(v), np.mean(v), np.array(v).argmax())
                        res3.append(np.max(v))
                    for k,v in d5.items():
                        print(k, np.max(v), np.mean(v), np.array(v).argmax())
                        res5.append(np.max(v))
                print("="*30)
                print(cid, labels[cid])
                d = {"clip%d" % j: [] for j in range(1, 5)}
                d3 = {"clip%d" % j: [] for j in range(1, 5)}
                d5 = {"clip%d" % j: [] for j in range(1, 5)}
            pred = y_pred[i]
            if "clip1" in npys[i]:
                d["clip1"].append(pred == cid)
                d3["clip1"].append(cid in inst_results[i].argsort()[-3:][::-1])
                d5["clip1"].append(cid in inst_results[i].argsort()[-5:][::-1])
            elif "clip2" in npys[i]:
                d["clip2"].append(pred == cid)
                d3["clip2"].append(cid in inst_results[i].argsort()[-3:][::-1])
                d5["clip2"].append(cid in inst_results[i].argsort()[-5:][::-1])
            elif "clip3" in npys[i]:
                d["clip3"].append(pred == cid)
                d3["clip3"].append(cid in inst_results[i].argsort()[-3:][::-1])
                d5["clip3"].append(cid in inst_results[i].argsort()[-5:][::-1])
            elif "clip4" in npys[i]:
                d["clip4"].append(pred == cid)
                d3["clip4"].append(cid in inst_results[i].argsort()[-3:][::-1])
                d5["clip4"].append(cid in inst_results[i].argsort()[-5:][::-1])
            y_pred_top5 = inst_results[i].argsort()[-5:][::-1]
            for k, pred_k in enumerate(y_pred_top5):
                print("%s: %s" % (k+1, labels[pred_k]))
            # if pred == cid:
            #     print(
            #         "    %s: ok" % npys[i].replace('../datasets/REAL/', ''),
            #         inst_results[i].argsort()[-10:][::-1]
            #     )
            # else:
            #     print(
            #         "    %s: %s" % (npys[i].replace('../datasets/REAL/', ''), labels[pred]),
            #         inst_results[i].argsort()[-10:][::-1]
            #     )
        else:
            for k,v in d.items():
                print(k, np.max(v), np.mean(v))
                res.append(np.max(v))
            for k,v in d3.items():
                print(k, np.max(v), np.mean(v))
                res3.append(np.max(v))
            for k,v in d5.items():
                print(k, np.max(v), np.mean(v))
                res5.append(np.max(v))
            print("="*30)
            print(len(res), np.mean(res))
            print(len(res3), np.mean(res3))
            print(len(res5), np.mean(res5))
    else:
        y_pred = np.array([[inst_results[i*4+j].argmax() for j in range(4)] for i in range(25)])
        y_pred_top5 = np.array([[inst_results[i*4+j].argsort()[-5:][::-1] for j in range(4)] for i in range(25)])
        y_true = np.array([[inst_targets[i*4+j] for j in range(4)] for i in range(25)])
        acc_class = (y_pred == y_true).mean(axis=1)
        acc_all = acc_class.mean(axis=0)
        print(y_pred)
        print(y_true)

        for i, cid in enumerate(y_true[:, 0]):
            print("%s: \t%.4f" % (labels[cid], acc_class[i]))
            for clip in range(4):
                print("clip:", clip)
                for k, pred_k in enumerate(y_pred_top5[i][clip]):
                    print("%s: %s" % (k+1, labels[pred_k]))
            # for pred in np.unique(y_pred[i]):
            #     if pred != cid:
            #         print("    *", labels[pred])
            print()

        print(acc_all)

    print('Loss {loss.avg:.4f}\t'
          'Acc@1 {top1.avg:.4f}\t'
          'Acc@3 {top3.avg:.4f}\t'
          'Acc@5 {top5.avg:.4f}\t'
          ''.format(
              loss=losses,
              top1=top1,
              top3=top3,
              top5=top5,
          ))



if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
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
    if opt.dataset != 'real':
        raise
    # with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    #     json.dump(vars(opt), opt_file, indent=2)
    if not opt.resume_path and not opt.init_path:
        raise

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    from torch.backends import cudnn
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu" if opt.no_cuda else "cuda")
    if not opt.no_cuda:
        criterion = criterion.to(device)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])

    if not opt.no_val:
        spatial_transform = Compose([
            ToTensor(opt.norm_value), norm_method,
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        if opt.compress == 'mask':
            spatio_temporal_transform = None
            temporal_transform = None
        elif opt.compress == 'avg':
            spatio_temporal_transform = Averaged()
        elif opt.compress == 'one':
            spatio_temporal_transform = OneFrame()
        elif opt.compress == 'spatial':
            spatial_transform = Compose([
                LowResolution(opt.spatial_compress_size, use_cv2=opt.use_cv2),
                ToTensor(opt.norm_value), norm_method,
            ])
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        else:
            spatio_temporal_transform = None
            temporal_transform = TemporalCenterCrop(opt.sample_duration)
        test_data = get_test_set(
            opt, spatial_transform, temporal_transform, target_transform,
            spatio_temporal_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test_logger = Logger(
            os.path.join(opt.result_path, 'test_.log'),
            ['top1', 'top3', 'top5'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # if not opt.no_train:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
    if opt.init_path:
        check = False
        print('initilize checkpoint {}'.format(opt.init_path))
        checkpoint = torch.load(opt.init_path)

        for i, (name, param) in enumerate(model.named_parameters()):
            for (prev_name, prev_param) in checkpoint['state_dict'].items():
                if name == prev_name:
                    param.data = prev_param.data
                    print('{}: {}({}) -> {}({})'.format(
                        i, prev_name, prev_param.shape, name, param.shape))
                    check = True
        if not check:
            raise

    model.to(device)
    test_eval(test_loader, model, criterion, opt, test_logger, device)
