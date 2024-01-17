import argparse
import torch
import os
import numpy as np
import dataset.crowd as crowd
from models import FusionCount
import time
import glob
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--weight_path', type=str, default='ckpt/sha/best_model.pth',
                        help='saved model path')
    parser.add_argument('--dataset', type=str, default='sha',
                        help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--save_path', default='', help='final image save path')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--backbone', default='vgg16')

    args = parser.parse_args()
    return args

def main(args):
    times = []
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # set vis gpu
    device = torch.device('cuda')

    model_path = args.weight_path
    crop_size = args.crop_size
    dataset = crowd.Crowd_(os.path.join('dataset', args.dataset, 'test'), crop_size, 8, method='val')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)
    
    model = FusionCount(args)
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []
    for step, (inputs, count, name) in enumerate(dataloader):
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = count[0].item() - torch.sum(outputs).item()

        if args.verbose:
            print(name, img_err, count[0].item(), torch.sum(outputs).item())
        image_errs.append(img_err)

        if args.save_path:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            original_image = cv2.imread(os.path.join('dataset', args.dataset, 'test', f'{name[0]}.jpg'))
            density_map = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

            density_map = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
            alpha = 0.6
            overlay = cv2.addWeighted(original_image, 1 - alpha, density_map, alpha, 0)
            cv2.putText(overlay, "Count:" + str(int(torch.sum(outputs).item())), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
            os.makedirs(os.path.join(args.save_path), exist_ok=True)
            cv2.imwrite(os.path.join(args.save_path, f'{name[0]}.jpg'), overlay)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print(f'{model_path} mse: {round(mse, 2)} mae: {round(mae, 2)}')

if __name__ == "__main__":
    args = get_args_parser()
    main(args)
