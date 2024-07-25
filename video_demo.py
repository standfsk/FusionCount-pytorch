import argparse
import torch
import os
import numpy as np
import dataset.crowd as crowd
from models import FusionCount
import time
import glob
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--gpu_id', default='0', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--weight_path', type=str, default='ckpt/sha_shb_qnrf/best_model.pth',
                    help='saved model path')
parser.add_argument('--video_name', default='video2.mp4', help='video')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

def video_to_image(pth, name):
    input_path = os.path.join(pth, name)
    output_folder = os.path.join(pth, f'{name.split(".")[0]}_original')
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    # total_frames = 450 if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 450 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_digits = len(str(total_frames))
    for frame_count in tqdm(range(total_frames)):
        ret, frame = cap.read()

        if not ret:
            break

        output_path = os.path.join(output_folder, f"frame_{frame_count + 1:0{num_digits}d}.jpg")
        resized_frame = cv2.resize(frame, ((1920, 1080)))
        cv2.imwrite(output_path, resized_frame)

    cap.release()

def image_to_video(pth, name):
    input_folder = os.path.join(pth, f'{name.split(".")[0]}_output')
    os.makedirs(input_folder, exist_ok=True)

    output_video_path = os.path.join(pth, f'{name.split(".")[0]}_output.mp4')
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    image_files.sort()

    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image_file in tqdm(image_files):
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')
    pth = 'result'
    if not os.path.exists(os.path.join(pth, f'{args.video_name.split(".")[0]}_original')):
        video_to_image(pth, args.video_name)
    else:
        print('frames already exist!!')

    model_path = args.weight_path
    crop_size = args.crop_size
    data_path = os.path.join(pth, f'{args.video_name.split(".")[0]}_original')

    dataset = crowd.Crowd_no(data_path, crop_size, 8, method='test')
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True)

    model = FusionCount(args)
    if args.data_type == 'fp32':
        model.to(device)
    elif args.data_type == 'fp16':
        model.half().to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    times = []
    for step, (inputs, name) in enumerate(dataloader):
        if args.data_type == 'fp32':
            inputs = inputs.to(device)
        elif args.data_type == 'fp16':
            inputs = inputs.half().to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            t1 = time.time()
            outputs = model(inputs)
            t2 = time.time()
            times.append(t2-t1)

        #print(f'{name[0]} pred_cnt: {np.sum(outputs)}')
        if args.save:
            vis_img = outputs[0, 0].cpu().numpy()
            # normalize density map values from 0 to 1, then map it to 0-255.
            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            density_map = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            # density_map = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)

            original_image = cv2.imread(os.path.join(data_path, name[0] + '.jpg'))
            density_map = cv2.resize(density_map, (original_image.shape[1], original_image.shape[0]))
            alpha = 0.6
            overlay = cv2.addWeighted(original_image, 1 - alpha, density_map, alpha, 0)

            os.makedirs(os.path.join(pth, f'{args.video_name.split(".")[0]}_output'), exist_ok=True)
            cv2.putText(overlay, f'predicted: {int(torch.sum(outputs).item())}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(pth, f'{args.video_name.split(".")[0]}_output', name[0] + '.jpg'), overlay)
            # print(f'{name[0]} pred_cnt: {int(torch.sum(outputs).item())}')

    # print(round(np.sum(times), 2), round(np.mean(times), 2))

    if args.save:
        image_to_video(pth, args.video_name)

if __name__ == "__main__":
    args.video_name = f'sample.mp4'
    args.data_type = 'fp32'
    args.save = True
    main(args)

