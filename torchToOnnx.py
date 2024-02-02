import argparse
import torch
import os
import numpy as np
import dataset.crowd as crowd
from models import FusionCount
import time
import onnx
import onnxruntime
from onnxsim import simplify

def get_args_parser():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='3', help='assign device')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--weight_path', type=str, default='ckpt/sha_shb_qnrf/best_model.pth',
                        help='saved model path')
    parser.add_argument('--backbone', default='vgg16')
    parser.add_argument('--save_path', default='')

    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.weight_path
    onnx_name = args.save_path
    dataset_name = 'UCF_CC_50'

    dataset = crowd.Crowd_(os.path.join('dataset', dataset_name, 'test'), args.crop_size, 8, method='test')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
    inputs, name = next(iter(dataloader))
    inputs = inputs.to(device)

    model = FusionCount(args)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    with torch.set_grad_enabled(False):
        # torch_output, _ = model(inputs)
        torch.onnx.export(model, inputs, onnx_name, opset_version=12)

    onnx_model = onnx.load(onnx_name)
    onnx_sim, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_sim, onnx_name)

if __name__ == "__main__":
    args = get_args_parser()
    main(args)
