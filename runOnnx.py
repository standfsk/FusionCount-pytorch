import onnxruntime
import os
import torch
import dataset.crowd as crowd
import cv2
import numpy as np
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--weight_path', type=str, default='fc.onnx',
                        help='saved model path')
    parser.add_argument('--backbone', default='vgg16')

    args = parser.parse_args()
    return args

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda')

    dataset_name = 'UCF_CC_50'
    dataset = crowd.Crowd_(os.path.join('dataset', dataset_name, 'test'), args.crop_size, 8, method='test')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
    inputs, name = next(iter(dataloader))
    inputs = inputs.to(device)

    ort_session = onnxruntime.InferenceSession(args.weight_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}

    ort_outs, _ = ort_session.run(None, ort_inputs)
    vis_img = ort_outs[0, 0]
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)

    original_image = cv2.imread(os.path.join('dataset', dataset_name, 'test', 'UCF_CC_50_00001.jpg'))

    density_map = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    density_map = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
    cv2.imwrite('dense.jpg', density_map)

    overlay_weight = 0.6
    overlay = cv2.addWeighted(original_image, 1 - overlay_weight, density_map, overlay_weight, 0)
    cv2.imwrite('final.jpg', overlay)
    print("onnx run success!!")


if __name__ == "__main__":
    args = get_args_parser()
    main(args)

