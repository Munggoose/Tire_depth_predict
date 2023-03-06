from xmlrpc.client import boolean
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision import transforms

from tqdm import tqdm 
from tqdm import trange
import torch.optim as optim 
import torch
from PIL import Image
from glob import glob
import pandas as pd

from torch.utils.mobile_optimizer import optimize_for_mobile
from datetime import datetime
import argparse
import yaml
import os
from main import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("-cfg",'--config',type=str,required=True, help="path to config file")
    # parser.add_argument("--data", type=str, required=True, help='path to dataset')
    parser.add_argument('--pt_path',type=str,help='path to model weight file')
    args = parser.parse_args()
    return args


def load_model(args):
    config = yaml.safe_load(open(args.config, "r"))
    model = build_model(config)
    # model = model
    model.load_state_dict(torch.load(args.pt_path)['model_state_dict'])
    model.eval()
    return model


def transform_mobile_format(model,example = None):
    
    if example is not None:
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module_optimized = optimize_for_mobile(traced_script_module)      
        traced_script_module_optimized._save_for_lite_interpreter("android_model.ptl")
        return 
        
    
    
    scripted_module = torch.jit.script(model)

    optimized_scripted_module = optimize_for_mobile(scripted_module)
    
    
    # Export full jit version model (not compatible with lite interpreter)    
    scripted_module.save("and_model.pt")
    # Export lite interpreter version model (compatible with lite interpreter)
    # scripted_module._save_for_lite_interpreter("sample_scripted_module_lite_interpreter.ptl")
    # # using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
    # optimized_scripted_module._save_for_lite_interpreter("Efficientnet_scripted_optimized.ptl")
    print('save script mode')


if __name__ == '__main__':
    # example = torch.randn((1,3,640,480))
    example = None
    model = torch.load('./outputs/sample.pt')
    model.eval()
    model.to('cpu')
    # output = model(example)

    print(model.training)
    transform_mobile_format(model,example)
