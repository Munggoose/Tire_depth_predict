# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import random
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
from models import build_model
from models.detr import SetCriterion
from models.matcher import build_matcher
import math
from tqdm import tqdm
from util.box_ops import rescale_bboxes
from util.visualizer import Visualizer
from util.metric import F1_score


CLASSES = ['4.0','5.5','4.5','1.5','7.0','N\A']



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # parser.add_argument('--num_queries', default=100, type=int,
    #                     help="Number of query slots")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=1, type=float)
    parser.add_argument('--giou_loss_coef', default=0, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='tire')
    parser.add_argument('--coco_path', type=str,default='./export/')
    parser.add_argument('--image_root',type=str, default='')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--n_class',default=6,type=int)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distribute', default=True,type=bool)
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
    model._modules['class_embed'] = nn.Linear(in_features=256, out_features= args.n_class+1,bias=True)
    model._modules['qeury_embed'] = nn.Embedding(args.num_queries,256)
    

    _, criterion, postprocessors = build_model(args)
    # model.to(device)

    # model = nn.DataParallel(model)    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1])
        model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    
    data_loader_train = DataLoader(dataset_train,
                                    collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                    batch_size=args.batch_size,drop_last=True,shuffle=True)
    
    # data_loader_val = DataLoader(dataset_val, batch_size=2, 
    #                                 drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers,shuffle=True)
    output_dir = Path(args.output_dir)
    
    
    # weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict = {'loss_ce': 5, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    
    losses = ['labels', 'boxes', 'cardinality']
    
    
    if args.masks:
        losses += ["masks"]
    
    
    # vis = Visualizer(model,args)
    model.cuda()
    criterion.cuda()
    best_loss = np.inf
    for epoch in tqdm(range(args.start_epoch, args.epochs),leave=False):
        # matcher = build_matcher(args)
        # criterion = SetCriterion(args.n_class, matcher=matcher, weight_dict=weight_dict,
        #                         eos_coef=args.eos_coef, losses=losses)
        
        
        train_loss = []
        class_losses = []
        val_loss = []
        #train
        model.train()
        criterion.train()
        for samples, targets in tqdm(data_loader_train):
            
            print(targets)
            exit()

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            class_loss = loss_dict_reduced_scaled['loss_ce'].item()

            class_losses.append(class_loss)
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
                        
            loss_value = losses_reduced_scaled.item()
            train_loss.append(loss_value)

            optimizer.zero_grad()
            losses.backward()

            # if max_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        lr_scheduler.step()
        class_loss_mean = np.array(class_losses).mean()
        mean_loss = np.array(train_loss).mean()
        
        print('train_loss ',mean_loss, ' class loss:', class_loss_mean)
        
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, './outputs/sample.pt')
        # if epoch > 100:
        #     pred_val = [] 
        #     targets_val = []
        
        #     model.eval()
        #     criterion.eval()

        #     for samples, targets in data_loader_val:
        #         samples = samples.to(device)
                
        #         #for f1-score
        #         val_target = ([t['labels'].tolist() for t in targets])
                
        #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #         outputs = model(samples)
                
        #         loss_dict = criterion(outputs, targets)
        #         weight_dict = criterion.weight_dict
        #         losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        #         # reduce losses over all GPUs for logging purposes
        #         loss_dict_reduced = utils.reduce_dict(loss_dict)

        #         loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
        #         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        #         loss_value = losses_reduced_scaled.item()
        #         #vis.visual_attention_weight(model,samples[0].unsquueze(0))
        #         val_loss.append(loss_value)
        #         probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]
                
        #         pred_val.extend(probas.to('cpu').argmax(-1).squeeze(0).tolist())
        #         targets_val.extend(val_target)


        #     # f1_score = F1_score(targets_val, pred_val)


        #     wandb_dict = {'train_loss':np.array(train_loss).mean(),
        #                     'val_loss': np.array(val_loss).mean()}
            
        #     if wandb_dict['val_loss'] < best_loss:
        #         best_loss = wandb_dict['val_loss']
        #         torch.save(model, './outputs/sample.pt')
            
            # vis.visual_log(wandb_dict)
        
        
            
if __name__=='__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)