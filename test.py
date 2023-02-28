
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import argparse
import random

import time
import datetime
import math

from transform_mobile import transform_mobile_format

CLASSES = ['4.0','5.5','4.5','1.5','7.0','N\A']


def get_args_parser():
    parser = argparse.ArgumentParser('Set Quantization')
    parser.add_argument('--quantization',action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    
    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False,  linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('fig1.png')


args = get_args_parser().parse_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

model = torch.load('./outputs/sample.pt')
model.to('cpu')
model.eval()

if args.quantization:
    # model.fuse_model()
    print('---------Start quantization---------------')
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # torch.ao.quantization.prepare(model, inplace=True)
    print(model.backbone[0].body._modules)


    model.backbone[0].body._modules = torch.quantization.fuse_modules(model.backbone[0].body, [['conv1','bn1'],['conv2','bn2']])
    model_prepared = torch.quantization.prepare(model)
    input_sample = torch.randn(4, 3, 640, 480)
    model_prepared(input_sample)
    model = torch.quantization.convert(model_prepared)
    # torch.ao.quantization.convert(model, inplace=True)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# transform_mobile_format(model)

if __name__=='__main__':
    imgs = glob("./sample_data/*.jpg")

    for im_path in imgs:
        im = Image.open(im_path)
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        # propagate through the model
        start = time.time()
        outputs = model(img)
        end = time.time()
        sec = end - start
        consum_t = datetime.timedelta(seconds=sec)
        print(consum_t)
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        
        keep = probas.max(-1).values == probas.max()

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        print(os.path.basename(im_path),CLASSES[probas[keep].argmax()])

    # plot_results(im, probas[keep], bboxes_scaled)