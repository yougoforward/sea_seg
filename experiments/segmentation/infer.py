###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os

import torch
import torchvision.transforms as transform

import encoding.utils as utils

from tqdm import tqdm

from torch.utils import data

from encoding.nn import BatchNorm
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model
from encoding.models import MultiEvalModule_whole as MultiEvalModule
from .option import Options


def test(args):
    # output folder
    outdir = args.save_folder
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = get_segmentation_dataset(args.dataset, split=args.split, mode=args.mode,
                                       transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated, multi_grid =args.multi_grid,
                                       stride =args.stride, 
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    evaluator = MultiEvalModule(model, testset.num_class, scales=scales, flip=args.ms).cuda()
    evaluator.eval()
    metric = utils.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0

    for i, (image, dst) in enumerate(tbar):
        if 'val' in args.mode:
            with torch.no_grad():
                predicts = evaluator.parallel_forward(image)
                metric.update(dst, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            with torch.no_grad():
                outputs = evaluator.parallel_forward(image[0])[0]
                # print(image)
                # print(outputs)
                # predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
                #             for output in outputs]
                correct, labeled = utils.batch_pix_accuracy(outputs, dst)
                total_correct += correct
                all_label += labeled
                img_pixAcc = 1.0 * correct / (np.spacing(1) + labeled)

                inter, union, area_pred, area_lab = utils.batch_intersection_union(outputs, dst, testset.num_class)
                total_label += area_lab
                total_inter += inter
                total_union += union

                class_pixAcc = 1.0 * inter / (np.spacing(1) + area_lab)
                class_IoU = 1.0 * inter / (np.spacing(1) + union)
                class_mIoU = class_IoU.mean()
                print("img pixAcc:", img_pixAcc)
                print("img Classes pixAcc:", class_pixAcc)
                print("img Classes IoU:", class_IoU)

            # for predict, impath in zip(predicts, dst):
            #     mask = utils.get_mask_pallete(predict, args.dataset)
            #     outname = os.path.splitext(impath)[0] + '.png'
            #     mask.save(os.path.join(outdir, outname))
    total_pixAcc = 1.0 * total_correct / (np.spacing(1) + all_label)
    pixAcc = 1.0 * total_inter / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()

    print("set pixAcc:", pixAcc)
    print("set Classes pixAcc:", pixAcc)
    print("set Classes IoU:", IoU)
    print("set mean IoU:", mIoU)

    # for i, (image, dst) in enumerate(tbar):
    #     if 'val' in args.mode:
    #         with torch.no_grad():
    #             predicts = evaluator.parallel_forward(image)
    #             metric.update(dst, predicts)
    #             pixAcc, mIoU = metric.get()
    #             tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    #     else:
    #         with torch.no_grad():
    #             outputs = evaluator.parallel_forward(image)
    #             predicts = [testset.make_pred(torch.max(output, 1)[1].cpu().numpy())
    #                         for output in outputs]
    #         for predict, impath in zip(predicts, dst):
    #             mask = utils.get_mask_pallete(predict, args.dataset)
    #             outname = os.path.splitext(impath)[0] + '.png'
    #             mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)


def batch_intersection_union(output, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(output, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union, area_pred, area_lab


def batch_pix_accuracy(output, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(output, 1)

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target)*(target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled