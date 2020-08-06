###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os

import torch
import torchvision.transforms as transform

import encoding.utils as utils

from PIL import Image

from encoding.nn import BatchNorm
from encoding.datasets import datasets
from encoding.models import get_model, get_segmentation_model
from encoding.models import MultiEvalModule_whole as MultiEvalModule

from .option import Options
import numpy as np

def test(args):
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
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
    num_classes = datasets[args.dataset.lower()].NUM_CLASS
    evaluator = MultiEvalModule(model, num_classes, scales=scales, flip=args.ms).cuda()
    evaluator.eval()

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        im_list = os.listdir(args.input_path)
        i=0
        for im in im_list:
            im_path = os.path.join(args.input_path, im)
            image = Image.open(im_path).convert('RGB')
            img = input_transform(image).unsqueeze(0)
            with torch.no_grad():
                output = evaluator.parallel_forward(img)[0]
                predict = torch.max(output, 1)[1].cpu().numpy()
            mask = utils.get_mask_pallete(predict, args.dataset)
            print(str(i)+'\t'+im)
            i=i+1

            mask =  np.array(mask).astype(np.uint8)
            image = np.array(image).astype(np.uint8)
            att = predict.squeeze()
            image[att==1] = image[att==1]*0.5+mask[att==1]*0.5
            mask = Image.fromarray(image)
            out_path = os.path.join(args.save_path, im)
            mask.save(out_path)
    else:
        img = input_transform(Image.open(args.input_path).convert('RGB')).unsqueeze(0)

        with torch.no_grad():
            output = evaluator.parallel_forward(img)[0]
            predict = torch.max(output, 1)[1].cpu().numpy()
        mask = utils.get_mask_pallete(predict, args.dataset)
        mask.save(args.save_path)



if __name__ == "__main__":
    option = Options()
    option.parser.add_argument('--input-path', type=str, required=True, help='path to read input image')
    option.parser.add_argument('--save-path', type=str, required=True, help='path to save output image')
    args = option.parse()

    torch.manual_seed(args.seed)

    test(args)
